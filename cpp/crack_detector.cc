#include "crack_detector.h"
#include "image_utils.h"
#include <rknn_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "image_utils.h"

#define CONF_THRESHOLD 0.7f
#define NMS_THRESHOLD 0.5f
#define MASK_THRESHOLD 0.5f
#define MIN_BOX_SIZE 30
#define MAX_BOX_SIZE 200

int init_crack_detector(const char* model_path, rknn_crack_detector_t* ctx)
{
    int ret;
    rknn_context rknn_ctx = 0;

    ret = rknn_init(&rknn_ctx, (void*)model_path, 0, 0, NULL);
    if (ret < 0) {
        printf("rknn_init failed! ret=%d\n", ret);
        return -1;
    }

    ctx->rknn_ctx = rknn_ctx;

    rknn_tensor_attr input_attr;
    memset(&input_attr, 0, sizeof(input_attr));
    input_attr.index = 0;
    ret = rknn_query(rknn_ctx, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    if (ret < 0) {
        printf("rknn_query input attr failed! ret=%d\n", ret);
        return -1;
    }

    ctx->model_width = input_attr.dims[2];
    ctx->model_height = input_attr.dims[1];
    ctx->model_channel = input_attr.dims[3];
    ctx->is_quant = (input_attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC);

    printf("Model input: %dx%dx%d, quant=%d\n",
           ctx->model_width, ctx->model_height, ctx->model_channel, ctx->is_quant);

    ctx->num_outputs = 2;

    for (int i = 0; i < ctx->num_outputs; i++) {
        rknn_tensor_attr output_attr;
        memset(&output_attr, 0, sizeof(output_attr));
        output_attr.index = i;
        ret = rknn_query(rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
        if (ret < 0) {
            printf("rknn_query output[%d] attr failed! ret=%d\n", i, ret);
            continue;
        }
        printf("Output[%d]: scale=%.4f, zp=%d, dims=[%d,%d,%d,%d], size=%d bytes\n",
               i, output_attr.scale, output_attr.zp,
               output_attr.dims[0], output_attr.dims[1], 
               output_attr.dims[2], output_attr.dims[3],
               output_attr.size);
        
        if (i == 1) {
            ctx->proto_channel = output_attr.dims[1];
            ctx->proto_height = output_attr.dims[2];
            ctx->proto_width = output_attr.dims[3];
            printf("Mask prototype: %dx%dx%d\n", ctx->proto_width, ctx->proto_height, ctx->proto_channel);
        }
    }

    return 0;
}

int release_crack_detector(rknn_crack_detector_t* ctx)
{
    if (ctx->rknn_ctx != 0) {
        rknn_destroy(ctx->rknn_ctx);
        ctx->rknn_ctx = 0;
    }
    return 0;
}

static float iou(box_t a, box_t b) {
    int inter_left = a.left > b.left ? a.left : b.left;
    int inter_top = a.top > b.top ? a.top : b.top;
    int inter_right = a.right < b.right ? a.right : b.right;
    int inter_bottom = a.bottom < b.bottom ? a.bottom : b.bottom;

    if (inter_left >= inter_right || inter_top >= inter_bottom) return 0.0f;

    int inter_area = (inter_right - inter_left) * (inter_bottom - inter_top);
    int area_a = (a.right - a.left) * (a.bottom - a.top);
    int area_b = (b.right - b.left) * (b.bottom - b.top);

    return (float)inter_area / (area_a + area_b - inter_area);
}

static int nms(crack_detect_result_list* results, float nms_threshold) {
    if (results->count <= 1) return results->count;

    int i, j;
    int keep[MAX_DETECTIONS] = {0};
    int keep_count = 0;

    for (i = 0; i < results->count; i++) {
        int flag = 1;
        for (j = 0; j < keep_count; j++) {
            if (iou(results->results[i].box, results->results[keep[j]].box) > nms_threshold) {
                flag = 0;
                break;
            }
        }
        if (flag) {
            keep[keep_count++] = i;
        }
    }

    crack_detect_result_t temp[MAX_DETECTIONS];
    for (i = 0; i < keep_count; i++) {
        temp[i] = results->results[keep[i]];
    }
    for (i = 0; i < keep_count; i++) {
        results->results[i] = temp[i];
    }
    results->count = keep_count;

    return keep_count;
}

static int cmp(const void* a, const void* b) {
    crack_detect_result_t* da = (crack_detect_result_t*)a;
    crack_detect_result_t* db = (crack_detect_result_t*)b;
    return (da->prop < db->prop) - (da->prop > db->prop);
}

static void process_mask(crack_detect_result_t* result, float* mask_proto, 
                        int proto_w, int proto_h, int proto_c,
                        int model_w, int model_h, int img_w, int img_h)
{
    int box_w = result->box.right - result->box.left;
    int box_h = result->box.bottom - result->box.top;
    
    if (box_w <= 0 || box_h <= 0) {
        printf("Invalid box size for mask processing\n");
        return;
    }

    float* mask = (float*)malloc(proto_w * proto_h * sizeof(float));
    if (!mask) {
        printf("malloc failed for mask\n");
        return;
    }

    for (int y = 0; y < proto_h; y++) {
        for (int x = 0; x < proto_w; x++) {
            float val = 0.0f;
            for (int c = 0; c < proto_c; c++) {
                val += result->mask_coeffs[c] * mask_proto[c * proto_h * proto_w + y * proto_w + x];
            }
            val = 1.0f / (1.0f + expf(-val));
            mask[y * proto_w + x] = val;
        }
    }

    int* mask_data = (int*)malloc(box_w * box_h * sizeof(int));
    if (!mask_data) {
        printf("malloc failed for mask_data\n");
        free(mask);
        return;
    }

    float scale_x = (float)proto_w / model_w;
    float scale_y = (float)proto_h / model_h;
    
    float box_left_model = (float)result->box.left * model_w / img_w;
    float box_top_model = (float)result->box.top * model_h / img_h;
    float box_right_model = (float)result->box.right * model_w / img_w;
    float box_bottom_model = (float)result->box.bottom * model_h / img_h;

    int mask_left = (int)(box_left_model * scale_x);
    int mask_top = (int)(box_top_model * scale_y);
    int mask_right = (int)(box_right_model * scale_x);
    int mask_bottom = (int)(box_bottom_model * scale_y);

    mask_left = mask_left < 0 ? 0 : mask_left;
    mask_top = mask_top < 0 ? 0 : mask_top;
    mask_right = mask_right > proto_w ? proto_w : mask_right;
    mask_bottom = mask_bottom > proto_h ? proto_h : mask_bottom;

    float inner_scale_x = (float)(mask_right - mask_left) / box_w;
    float inner_scale_y = (float)(mask_bottom - mask_top) / box_h;

    for (int y = 0; y < box_h; y++) {
        for (int x = 0; x < box_w; x++) {
            int mask_x = mask_left + (int)(x * inner_scale_x);
            int mask_y = mask_top + (int)(y * inner_scale_y);
            
            if (mask_x >= 0 && mask_x < proto_w && mask_y >= 0 && mask_y < proto_h) {
                mask_data[y * box_w + x] = (mask[mask_y * proto_w + mask_x] > MASK_THRESHOLD) ? 1 : 0;
            } else {
                mask_data[y * box_w + x] = 0;
            }
        }
    }

    result->mask_data = mask_data;
    result->mask_width = box_w;
    result->mask_height = box_h;

    free(mask);
}

int inference_crack_detector(rknn_crack_detector_t* ctx, image_buffer_t* img, crack_detect_result_list* results)
{
    int ret;

    int img_width = img->width;
    int img_height = img->height;
    int model_width = ctx->model_width;
    int model_height = ctx->model_height;

    float scale_x = (float)img_width / model_width;
    float scale_y = (float)img_height / model_height;
    
    printf("Input image: %dx%d, model: %dx%d, scale_x=%.4f, scale_y=%.4f\n", 
           img_width, img_height, model_width, model_height, scale_x, scale_y);

    unsigned char* input_buf = (unsigned char*)malloc(model_width * model_height * 3);
    if (!input_buf) {
        printf("malloc failed for input_buf\n");
        return -1;
    }

    memset(input_buf, 114, model_width * model_height * 3);

    if (img->format == IMAGE_FORMAT_RGB888) {
        printf("Input format: RGB888\n");
        
        // YOLO官方预处理：保持宽高比缩放 + letterbox填充
        float r = (float)model_width / img_width < (float)model_height / img_height ? 
                  (float)model_width / img_width : (float)model_height / img_height;
        
        int resize_w = (int)(img_width * r);
        int resize_h = (int)(img_height * r);
        
        int pad_w = (model_width - resize_w) / 2;
        int pad_h = (model_height - resize_h) / 2;
        
        printf("Resize: %dx%d -> %dx%d, pad: (%d, %d)\n", 
               img_width, img_height, resize_w, resize_h, pad_w, pad_h);
        
        // 双线性插值缩放并填充
        for (int dst_y = 0; dst_y < model_height; dst_y++) {
            for (int dst_x = 0; dst_x < model_width; dst_x++) {
                // 计算源图像坐标
                float src_x = (dst_x - pad_w) / r;
                float src_y = (dst_y - pad_h) / r;
                
                // 检查是否在有效区域内
                if (src_x < 0 || src_x >= img_width || src_y < 0 || src_y >= img_height) {
                    // 填充区域，使用114
                    input_buf[dst_y * model_width * 3 + dst_x * 3] = 114;
                    input_buf[dst_y * model_width * 3 + dst_x * 3 + 1] = 114;
                    input_buf[dst_y * model_width * 3 + dst_x * 3 + 2] = 114;
                    continue;
                }
                
                // 双线性插值
                int x0 = (int)src_x;
                int y0 = (int)src_y;
                int x1 = x0 + 1;
                int y1 = y0 + 1;
                
                if (x1 >= img_width) x1 = img_width - 1;
                if (y1 >= img_height) y1 = img_height - 1;
                
                float fx = src_x - x0;
                float fy = src_y - y0;
                
                int dst_idx = dst_y * model_width * 3 + dst_x * 3;
                
                for (int c = 0; c < 3; c++) {
                    float v0 = (1 - fx) * img->virt_addr[y0 * img_width * 3 + x0 * 3 + c] + fx * img->virt_addr[y0 * img_width * 3 + x1 * 3 + c];
                    float v1 = (1 - fx) * img->virt_addr[y1 * img_width * 3 + x0 * 3 + c] + fx * img->virt_addr[y1 * img_width * 3 + x1 * 3 + c];
                    input_buf[dst_idx + c] = (unsigned char)((1 - fy) * v0 + fy * v1);
                }
            }
        }
        
        // 保存缩放和填充参数用于后续坐标映射
        ctx->scale = r;
        ctx->pad_w = pad_w;
        ctx->pad_h = pad_h;
    } else {
        printf("Unsupported image format: %d\n", img->format);
        free(input_buf);
        return -1;
    }

    rknn_input inputs[1];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = input_buf;
    inputs[0].size = model_width * model_height * 3;
    inputs[0].pass_through = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    ret = rknn_inputs_set(ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        printf("rknn_inputs_set failed! ret=%d\n", ret);
        free(input_buf);
        return -1;
    }

    ret = rknn_run(ctx->rknn_ctx, NULL);
    if (ret < 0) {
        printf("rknn_run failed! ret=%d\n", ret);
        free(input_buf);
        return -1;
    }

    rknn_output outputs[2];
    memset(outputs, 0, sizeof(outputs));
    outputs[0].want_float = 1;
    outputs[0].is_prealloc = 0;
    outputs[1].want_float = 1;
    outputs[1].is_prealloc = 0;

    ret = rknn_outputs_get(ctx->rknn_ctx, 2, outputs, NULL);
    if (ret < 0) {
        printf("rknn_outputs_get failed! ret=%d\n", ret);
        free(input_buf);
        return -1;
    }

    float* det_output = (float*)outputs[0].buf;
    int det_size = outputs[0].size;
    
    float* mask_proto = (float*)outputs[1].buf;
    int mask_proto_h = ctx->proto_height;
    int mask_proto_w = ctx->proto_width;
    int mask_proto_c = ctx->proto_channel;

    printf("Detection output size: %d bytes\n", det_size);
    printf("Mask proto: %dx%dx%d\n", mask_proto_w, mask_proto_h, mask_proto_c);

    results->count = 0;

    int num_channels = 37;
    int num_boxes = det_size / (num_channels * sizeof(float));

    printf("Num channels: %d, Num boxes: %d\n", num_channels, num_boxes);

    int debug_count = 0;
    for (int i = 0; i < num_boxes && results->count < MAX_DETECTIONS; i++) {
        float x_center = det_output[i * num_channels + 0];
        float y_center = det_output[i * num_channels + 1];
        float width = det_output[i * num_channels + 2];
        float height = det_output[i * num_channels + 3];
        float conf = det_output[i * num_channels + 4];
        float raw_conf = conf;

        conf = conf / 300.0f;
        conf = 1.0f / (1.0f + expf(-conf));

        if (debug_count < 10) {
            printf("Debug: box[%d] = (%.3f, %.3f, %.3f, %.3f, raw_conf=%.3f, conf=%.3f)\n", 
                   i, x_center, y_center, width, height, raw_conf, conf);
            debug_count++;
        }

        if (conf > CONF_THRESHOLD) {
            float left = x_center - width / 2;
            float top = y_center - height / 2;
            float right = left + width;
            float bottom = top + height;

            // letterbox逆变换：先减去填充，再除以缩放因子
            float r = ctx->scale;
            int pad_w = ctx->pad_w;
            int pad_h = ctx->pad_h;
            
            left = (left - pad_w) / r;
            top = (top - pad_h) / r;
            right = (right - pad_w) / r;
            bottom = (bottom - pad_h) / r;

            int box_left = (int)(left + 0.5f);
            int box_top = (int)(top + 0.5f);
            int box_right = (int)(right + 0.5f);
            int box_bottom = (int)(bottom + 0.5f);

            box_left = box_left < 0 ? 0 : box_left;
            box_top = box_top < 0 ? 0 : box_top;
            box_right = box_right > img_width ? img_width : box_right;
            box_bottom = box_bottom > img_height ? img_height : box_bottom;

            int w = box_right - box_left;
            int h = box_bottom - box_top;

            if (w > MIN_BOX_SIZE && h > MIN_BOX_SIZE && w < MAX_BOX_SIZE && h < MAX_BOX_SIZE) {
                results->results[results->count].box.left = box_left;
                results->results[results->count].box.top = box_top;
                results->results[results->count].box.right = box_right;
                results->results[results->count].box.bottom = box_bottom;
                results->results[results->count].prop = conf;
                results->results[results->count].cls_id = 0;
                
                for (int j = 0; j < MASK_COEF_COUNT; j++) {
                    results->results[results->count].mask_coeffs[j] = det_output[i * num_channels + 5 + j];
                }
                
                process_mask(&results->results[results->count], mask_proto,
                           mask_proto_w, mask_proto_h, mask_proto_c,
                           model_width, model_height, img_width, img_height);
                results->count++;
            }
        }
    }

    printf("Before NMS: %d detections\n", results->count);

    qsort(results->results, results->count, sizeof(crack_detect_result_t), cmp);
    nms(results, NMS_THRESHOLD);

    printf("After NMS: %d detections\n", results->count);

    for (int i = 0; i < results->count && i < 10; i++) {
        printf("Detection %d: box=(%d,%d,%d,%d), conf=%.3f, mask=%dx%d\n",
               i,
               results->results[i].box.left,
               results->results[i].box.top,
               results->results[i].box.right,
               results->results[i].box.bottom,
               results->results[i].prop,
               results->results[i].mask_width,
               results->results[i].mask_height);
    }

    rknn_outputs_release(ctx->rknn_ctx, 2, outputs);
    free(input_buf);

    return 0;
}

void release_detection_results(crack_detect_result_list* results)
{
    for (int i = 0; i < results->count; i++) {
        if (results->results[i].mask_data != NULL) {
            free(results->results[i].mask_data);
            results->results[i].mask_data = NULL;
        }
    }
    results->count = 0;
}
#ifndef _CRACK_DETECTOR_H_
#define _CRACK_DETECTOR_H_

#include <stdint.h>
#include <stdbool.h>
#include "common.h"
#include "rknn_api.h"

#define MASK_COEF_COUNT 32
#define MAX_DETECTIONS 200

typedef struct {
    rknn_context rknn_ctx;
    int model_width;
    int model_height;
    int model_channel;
    bool is_quant;
    int num_boxes;
    float output_scale;
    int32_t output_zp;
    int proto_width;
    int proto_height;
    int proto_channel;
    int num_outputs;
    float scale;
    int pad_w;
    int pad_h;
} rknn_crack_detector_t;

typedef struct {
    int left;
    int top;
    int right;
    int bottom;
} box_t;

typedef struct {
    box_t box;
    float prop;
    int cls_id;
    float mask_coeffs[MASK_COEF_COUNT];
    int* mask_data;
    int mask_width;
    int mask_height;
} crack_detect_result_t;

typedef struct {
    int count;
    crack_detect_result_t results[MAX_DETECTIONS];
} crack_detect_result_list;

int init_crack_detector(const char* model_path, rknn_crack_detector_t* ctx);
int release_crack_detector(rknn_crack_detector_t* ctx);
int inference_crack_detector(rknn_crack_detector_t* ctx, image_buffer_t* img, crack_detect_result_list* results);
void release_detection_results(crack_detect_result_list* results);

#endif
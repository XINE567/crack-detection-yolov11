#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/time.h>

#include "crack_detector.h"
#include "image_utils.h"
#include "image_drawing.h"
#include "file_utils.h"

#define LABEL_NAME "crack"

static volatile int quit = 0;

static void sig_handler(int signo)
{
    if (signo == SIGINT) {
        quit = 1;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path> [image_path]\n", argv[0]);
        printf("  model_path: Path to .rknn model file\n");
        printf("  image_path: Path to test image (optional)\n");
        return -1;
    }

    const char* model_path = argv[1];
    const char* image_path = argc > 2 ? argv[2] : NULL;

    signal(SIGINT, sig_handler);

    int ret;
    rknn_crack_detector_t ctx;
    memset(&ctx, 0, sizeof(rknn_crack_detector_t));

    printf("Initializing RKNN model...\n");
    ret = init_crack_detector(model_path, &ctx);
    if (ret != 0) {
        printf("init_crack_detector fail! ret=%d\n", ret);
        return -1;
    }

    if (image_path != NULL) {
        image_buffer_t src_image;
        memset(&src_image, 0, sizeof(image_buffer_t));

        printf("Reading image from file: %s\n", image_path);
        ret = read_image(image_path, &src_image);
        if (ret != 0) {
            printf("read_image fail! ret=%d\n", ret);
            return -1;
        }

        crack_detect_result_list results;
        memset(&results, 0, sizeof(results));
        
        ret = inference_crack_detector(&ctx, &src_image, &results);
        if (ret != 0) {
            printf("inference fail! ret=%d\n", ret);
            return -1;
        }

        printf("Detected %d cracks\n", results.count);
        for (int i = 0; i < results.count; i++) {
            crack_detect_result_t* det = &results.results[i];
            printf("%s @ (%d, %d, %d, %d) %.3f\n",
                   LABEL_NAME, det->box.left, det->box.top,
                   det->box.right, det->box.bottom, det->prop);

            if (det->mask_data != NULL && det->mask_width > 0 && det->mask_height > 0) {
                for (int y = 0; y < det->mask_height; y++) {
                    for (int x = 0; x < det->mask_width; x++) {
                        int mask_val = det->mask_data[y * det->mask_width + x];
                        if (mask_val > 0) {
                            int img_x = det->box.left + x;
                            int img_y = det->box.top + y;
                            if (img_x >= 0 && img_x < src_image.width && 
                                img_y >= 0 && img_y < src_image.height) {
                                int idx = img_y * src_image.width * 3 + img_x * 3;
                                src_image.virt_addr[idx] = 0;
                                src_image.virt_addr[idx + 1] = 255;
                                src_image.virt_addr[idx + 2] = 0;
                            }
                        }
                    }
                }
            }

            draw_rectangle(&src_image, det->box.left, det->box.top,
                          det->box.right - det->box.left, det->box.bottom - det->box.top,
                          COLOR_RED, 2);

            char text[256];
            sprintf(text, "%s %.1f%%", LABEL_NAME, det->prop * 100);
            draw_text(&src_image, text, det->box.left, det->box.top - 10,
                     COLOR_GREEN, 16);
        }

        char output_path[256];
        sprintf(output_path, "output_%s", image_path);
        write_image(output_path, &src_image);
        printf("Output saved to: %s\n", output_path);

        release_detection_results(&results);
    } else {
        printf("Please provide an image path for testing.\n");
        printf("Example: %s model/crack_detector.rknn test.jpg\n", argv[0]);
    }

    release_crack_detector(&ctx);
    printf("Done.\n");
    return 0;
}
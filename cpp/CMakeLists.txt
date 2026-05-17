cmake_minimum_required(VERSION 3.10)

project(rknn_crack_detector)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 99)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(RKNN_TOOLCHAIN_PATH /usr)
set(CMAKE_C_COMPILER ${RKNN_TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${RKNN_TOOLCHAIN_PATH}/bin/aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

set(RKNN_SDK_PATH /home/yp/rk3588_linux_sdk/RK3588/inference/3-atk_yolov5s_recognize/cpp/3rdparty/rknpu2)
set(LIBRGA_PATH /home/yp/rk3588_linux_sdk/RK3588/inference/3-atk_yolov5s_recognize/cpp/3rdparty/librga)
set(LIBJPEG_PATH /home/yp/rk3588_linux_sdk/RK3588/inference/3-atk_yolov5s_recognize/cpp/3rdparty/jpeg_turbo)
set(STB_PATH /home/yp/rk3588_linux_sdk/RK3588/inference/3-atk_yolov5s_recognize/cpp/3rdparty/stb_image)

include_directories(
    ${RKNN_SDK_PATH}/include
    ${LIBRGA_PATH}/include
    ${LIBJPEG_PATH}/include
    ${STB_PATH}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/utils
)

link_directories(
    ${RKNN_SDK_PATH}/Linux/aarch64
    ${LIBRGA_PATH}/Linux/aarch64
    ${LIBJPEG_PATH}/Linux/aarch64
)

set(LIBRGA ${LIBRGA_PATH}/Linux/aarch64/librga.so)
set(LIBJPEG ${LIBJPEG_PATH}/Linux/aarch64/libturbojpeg.a)

find_package(Threads REQUIRED)

add_library(imageutils STATIC
    ${PROJECT_SOURCE_DIR}/utils/image_utils.c
)
target_include_directories(imageutils PUBLIC
    ${PROJECT_SOURCE_DIR}/utils
    ${LIBJPEG_PATH}/include
    ${LIBRGA_PATH}/include
    ${STB_PATH}
)
target_link_libraries(imageutils ${LIBJPEG} ${LIBRGA} Threads::Threads)

add_library(fileutils STATIC
    ${PROJECT_SOURCE_DIR}/utils/file_utils.c
)
target_include_directories(fileutils PUBLIC ${PROJECT_SOURCE_DIR}/utils)

add_library(imagedrawing STATIC
    ${PROJECT_SOURCE_DIR}/utils/image_drawing.c
)
target_include_directories(imagedrawing PUBLIC ${PROJECT_SOURCE_DIR}/utils)

add_executable(rknn_crack_detector
    main.cc
    crack_detector.cc
)

target_link_libraries(rknn_crack_detector
    imageutils
    fileutils
    imagedrawing
    ${RKNN_SDK_PATH}/Linux/aarch64/librknnrt.so
    m
    Threads::Threads
)

install(TARGETS rknn_crack_detector DESTINATION bin)

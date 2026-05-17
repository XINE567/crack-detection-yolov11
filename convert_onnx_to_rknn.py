#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO11-seg ONNX转RKNN脚本（适配RK3588）
支持裂纹检测模型
"""

import argparse
import os
import sys
from rknn.api import RKNN


def convert_onnx_to_rknn(args):
    rknn = RKNN()
    
    print("[INFO] 配置模型参数...")
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform='rk3588',
        optimization_level=3,
        output_optimize=True
    )
    
    print(f"[INFO] 加载ONNX模型: {args.onnx_path}")
    ret = rknn.load_onnx(
        model=args.onnx_path
    )
    
    if ret != 0:
        print(f"[ERROR] 加载模型失败! ret={ret}")
        sys.exit(ret)
    
    print("[INFO] 构建RKNN模型...")
    ret = rknn.build(
        do_quantization=args.quantize,
        dataset=args.calib_data
    )
    
    if ret != 0:
        print(f"[ERROR] 构建模型失败! ret={ret}")
        sys.exit(ret)
    
    print(f"[INFO] 导出RKNN模型: {args.output_path}")
    ret = rknn.export_rknn(args.output_path)
    
    if ret != 0:
        print(f"[ERROR] 导出模型失败! ret={ret}")
        sys.exit(ret)
    
    if args.test:
        print("[INFO] 测试推理...")
        import cv2
        import numpy as np
        
        ret = rknn.init_runtime(target='rk3588')
        if ret != 0:
            print(f"[ERROR] 初始化运行时失败! ret={ret}")
            sys.exit(ret)
        
        test_img = cv2.imread(args.test_img)
        if test_img is None:
            print(f"[ERROR] 无法加载测试图像: {args.test_img}")
            sys.exit(-1)
        
        test_img = cv2.resize(test_img, (640, 640))
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = np.transpose(test_img, (2, 0, 1))
        test_img = test_img.astype(np.float32) / 255.0
        test_img = np.expand_dims(test_img, axis=0)
        
        outputs = rknn.inference(inputs=[test_img])
        print(f"[INFO] 输出形状: {outputs[0].shape}")
        print(f"[INFO] 输出范围: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
        
        rknn.release()
    
    print("[INFO] 转换完成!")


def main():
    parser = argparse.ArgumentParser(description='YOLO11-seg ONNX to RKNN Converter')
    parser.add_argument('--onnx_path', required=True, help='ONNX模型路径')
    parser.add_argument('--output_path', required=True, help='输出RKNN模型路径')
    parser.add_argument('--calib_data', default='calib.txt', help='校准数据集文件')
    parser.add_argument('--quantize', action='store_true', default=True, help='是否量化')
    parser.add_argument('--test', action='store_true', help='是否测试推理')
    parser.add_argument('--test_img', default='test.jpg', help='测试图像路径')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.onnx_path):
        print(f"[ERROR] ONNX模型不存在: {args.onnx_path}")
        sys.exit(-1)
    
    if args.quantize and not os.path.exists(args.calib_data):
        print(f"[ERROR] 校准数据文件不存在: {args.calib_data}")
        sys.exit(-1)
    
    convert_onnx_to_rknn(args)


if __name__ == '__main__':
    main()
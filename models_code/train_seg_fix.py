from ultralytics import YOLO

# 加载优化后的模型
model = YOLO('D:/code/yolov11/runs/crack_optimized/finetune_v1/weights/best.pt')

results = model.train(
    data='D:/crack-seg/data.yaml',
    imgsz=640,
    batch=8,
    epochs=20,
    device=0,
    workers=0,

    # 关键：降低学习率，精细调整
    lr0=0.0005,  # 比之前的0.005更低10倍
    lrf=0.0005,

    # 关闭/降低影响边缘的增强
    hsv_v=0.2,  # 降低明度增强（原0.6）
    scale=0.4,  # 降低尺度抖动（原0.8）
    mixup=0.0,  # 关闭混合增强
    copy_paste=0.0,  # 关闭复制粘贴

    # 保持后处理优化
    iou=0.6,
    conf=0.20,

    # 输出
    project='runs/crack_fix',
    name='fix_seg',
    exist_ok=True,
)

print("✅ 分割修复完成")
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载 crack_v2 的模型作为起点
    model = YOLO('runs/crack_final/crack_v2/weights/best.pt')

    results = model.train(
        data='D:/crack_fix-seg/data.yaml',
        imgsz=640,
        batch=8,
        epochs=50,
        device=0,
        workers=0,

        # 关键：极低学习率，只微调不破坏
        lr0=0.0005,  # 比默认 0.01 低 20 倍
        lrf=0.0005,
        warmup_epochs=0,  # 不预热，直接开始

        # 数据增强（只开对分割有帮助的）
        hsv_v=0.1,  # 轻微亮度变化，不破坏边缘
        scale=0.3,  # 轻微缩放
        fliplr=0.5,
        mosaic=0.0,  # 关闭，避免破坏边缘
        copy_paste=0.0,  # 关闭

        # 后处理
        iou=0.70,

        # 输出
        project='runs/crack_v3',
        name='crack_v3',
        exist_ok=True,
    )

    print("✅ 分割精度微调完成")
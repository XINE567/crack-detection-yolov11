from ultralytics import YOLO

if __name__ == '__main__':
    # fix_seg 的分割能力强 + 新数据集干净
    model = YOLO('runs/crack_fix/fix_seg/weights/best.pt')

    results = model.train(
        data='D:/crack_fix-seg/data.yaml',  # 新数据集
        imgsz=640,
        batch=8,
        epochs=80,
        device=0,
        workers=0,

        lr0=0.001,  # 中等学习率

        # 适度增强
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,

        iou=0.65,

        project='runs/crack_fix2',
        name='fix_seg2',
        exist_ok=True,
    )
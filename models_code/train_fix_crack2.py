from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolo11s-seg.pt')

    results = model.train(
        data='D:/crack_fix-seg/data.yaml',
        imgsz=640,
        batch=8,
        epochs=100,
        device=0,
        workers=0,

        scale=0.6,
        fliplr=0.5,
        mosaic=0,

        iou=0.7,

        project='runs/crack_2',
        name='crack_v2',
        exist_ok=True,
    )

    print("✅ 训练完成")
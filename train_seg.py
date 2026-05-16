# train_seg.py
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载分割模型
    model = YOLO('yolo11s-seg.pt')

    # 开始训练
    results = model.train(
        data=r'D:\crack-seg\data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        patience=20,
        save=True,
        project='runs/crack_seg',
        name='train6',
    )

    print("✅ 训练完成！")
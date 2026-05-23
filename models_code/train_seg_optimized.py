# train_optimized.py - 裂纹检测模型优化训练
from ultralytics import YOLO

# 加载已训练好的模型（作为起点进行微调）
model = YOLO('D:/code/yolov11/runs/crack_seg/train6/weights/best.pt')

# 开始训练
results = model.train(
    data='D:/crack-seg/data.yaml',
    imgsz=640,  # 输入尺寸（8GB显存稳定）
    batch=8,  # 批次大小
    epochs=40,  # 训练轮数
    device=0,  # GPU
    workers=0,  # 数据加载线程

    # 学习率（微调用低学习率）
    lr0=0.005,  # 初始学习率
    lrf=0.005,  # 最终学习率

    # 数据增强（暗光/小目标/脏污优化）
    hsv_v=0.6,  # 明度增强（模拟暗光）
    scale=0.8,  # 尺度抖动（小目标）
    mixup=0.3,  # 混合增强（泛化）
    copy_paste=0.5,  # 复制粘贴（增加样本）
    iou=0.6,  # NMS阈值（减少重复框）
    conf=0.20,  # 置信度阈值（捕获暗光裂纹）
    dropout=0.1,  # dropout（防过拟合）

    # 输出配置
    project='runs/crack_seg_optimized',
    name='crack_seg_optimized',
    exist_ok=True,
)

print("✅ 训练完成")
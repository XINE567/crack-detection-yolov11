# YOLO11-seg 裂纹检测 RKNN 部署包

## 功能特点

- 支持 **YOLO11-seg** 实例分割模型
- 输出裂纹的**精确形状掩码**（不只是边界框）
- 支持边界框 + 分割掩码叠加显示
- 适配 RK3588 开发板

## 项目结构

```
crack_detector_standalone/
├── convert_onnx_to_rknn.py    # ONNX转RKNN脚本
├── calib.txt                   # 校准数据集配置
├── README.md                   # 使用说明
├── model/                      # 模型目录（放您的ONNX和RKNN模型）
├── dataset/                    # 校准图像目录
└── cpp/
    ├── main.cc                 # 主入口文件
    ├── crack_detector.cc       # 推理实现（含掩码生成）
    ├── crack_detector.h        # 头文件
    ├── CMakeLists.txt          # 编译配置
    └── build/                  # 编译输出目录
```

## 模型输出格式

- **输出0 (检测头)**: `[1, 37, 8400]`
  - 通道 0-3: 边界框坐标 (cx, cy, w, h)
  - 通道 4: 置信度
  - 通道 5-36: 32个掩码系数

- **输出1 (原型掩码)**: `[1, 32, H, W]`
  - 32个原型掩码图

- **最终掩码**: `掩码系数 × 原型掩码 → Sigmoid → 精确裂纹形状`

## 使用步骤

### 第一步：准备模型和数据

1. 将您的 YOLO11-seg ONNX 模型放入 `model/` 目录
2. 在 `dataset/` 目录放入 100-500 张校准图像
3. 更新 `calib.txt` 添加图像路径

### 第二步：模型转换

```bash
python convert_onnx_to_rknn.py \
    --onnx_path model/your_model.onnx \
    --output_path model/crack_detector.rknn \
    --calib_data calib.txt \
    --quantize
```

### 第三步：交叉编译

```bash
cd cpp/build
cmake ..
make -j4
```

### 第四步：部署到开发板

```bash
# 复制可执行文件
scp rknn_crack_detector root@rk3588:/root/

# 复制模型
scp ../model/crack_detector.rknn root@rk3588:/root/model/
```

### 第五步：运行测试

```bash
# 在开发板上运行
cd /root
./rknn_crack_detector model/crack_detector.rknn
```

## 配置说明

### 检测阈值

在 `cpp/crack_detector.cc` 中修改：
```cpp
#define CONF_THRESHOLD 0.5f  // 置信度阈值
```

### 掩码透明度

在 `cpp/main.cc` 中修改：
```cpp
#define MASK_ALPHA 0.5  // 掩码叠加透明度 (0.0-1.0)
```

### 摄像头设备号

在 `cpp/main.cc` 中修改：
```cpp
cv::VideoCapture cap(0);  // 设备号
```

## 注意事项

1. 确保 ONNX 模型输入尺寸为 640×640
2. 校准图像建议与模型输入尺寸一致
3. 量化模型需要至少 100 张校准图像
4. 编译前确保交叉编译工具链路径正确

## 显示效果

```
┌─────────────────────────────────┐
│  ┌───────────────┐              │
│  │   精确裂纹形状 │  ← 绿色掩码  │
│  │  ████████████ │              │
│  └───────────────┘  ← 红色边框  │
│                                 │
│  crack 95.0%      ← 置信度标签  │
└─────────────────────────────────┘
```
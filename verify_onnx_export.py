import onnx
import onnxruntime as ort
import numpy as np

# 加载ONNX模型
model = onnx.load('model/best.onnx')

# 检查模型结构
print("=" * 60)
print("ONNX Model Analysis")
print("=" * 60)

# 检查输入
print("\n1. Inputs:")
for input in model.graph.input:
    print(f"   Name: {input.name}")
    shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
    print(f"   Shape: {shape}")

# 检查输出
print("\n2. Outputs:")
for output in model.graph.output:
    print(f"   Name: {output.name}")
    shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
    print(f"   Shape: {shape}")

# 检查节点数量
print(f"\n3. Number of nodes: {len(model.graph.node)}")

# 检查opset版本
print(f"\n4. Opset imports:")
for imp in model.opset_import:
    print(f"   Domain: {imp.domain}, Version: {imp.version}")

# 验证模型
try:
    onnx.checker.check_model(model)
    print("\n5. Model validation: PASSED")
except Exception as e:
    print(f"\n5. Model validation: FAILED - {e}")

# 测试推理
print("\n6. Testing inference...")
session = ort.InferenceSession('model/best.onnx', providers=['CPUExecutionProvider'])

# 创建随机输入
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"   Input name: {input_name}")
print(f"   Input shape: {input_shape}")

# 测试推理
random_input = np.random.randn(*input_shape).astype(np.float32)
outputs = session.run(None, {input_name: random_input})

print(f"\n7. Output shapes:")
for i, output in enumerate(outputs):
    print(f"   Output[{i}]: {output.shape}")

# 分析输出范围
print("\n8. Output analysis:")
for i, output in enumerate(outputs):
    print(f"   Output[{i}]: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)

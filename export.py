import torch
import onnx

# load your model
model = Your_Model()
model.eval()
input = torch.randn(16, 3, 224, 224)

torch.onnx.export(model, input, "model.onnx", input_names=["input"], opset_version=13)
model = onnx.load("model.onnx")

# 转换权重类型（将 INT64 转换为 INT32）
for tensor in model.graph.initializer:
    if tensor.data_type == onnx.TensorProto.INT64:
        tensor.data_type = onnx.TensorProto.INT32
onnx.save(model, "model.onnx")
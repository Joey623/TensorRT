import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

# 加载 TensorRT 引擎
def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    return trt_runtime.deserialize_cuda_engine(engine_data)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(TRT_LOGGER)
engine = load_engine(runtime, "model.trt")


# 创建推理上下文
context = engine.create_execution_context()

# 分配输入和输出的设备内存
input_shape = (16, 3, 224, 224)
input_nbytes = np.prod(input_shape) * np.float32().itemsize
output_shape = (16, engine.get_binding_shape(1)[1])
output_nbytes = np.prod(output_shape) * np.float32().itemsize

# d_input = cuda.mem_alloc(input_nbytes)
# d_output = cuda.mem_alloc(output_nbytes)
# 确保使用 Python 内置的 int 类型而不是 numpy.int64
d_input = cuda.mem_alloc(int(input_nbytes))  # 转换为 int 类型
d_output = cuda.mem_alloc(int(output_nbytes))  # 转换为 int 类型

# 创建流
stream = cuda.Stream()

# 准备输入数据
input_data = np.random.randn(*input_shape).astype(np.float32)
cuda.memcpy_htod_async(d_input, input_data, stream)

# 推理速度测试
iterations = 100  # 测试次数
start_time = time.time()

for _ in range(iterations):
    # 执行推理
    context.execute_async_v2([int(d_input), int(d_output)], stream.handle)
    stream.synchronize()  # 等待当前流完成

end_time = time.time()
inference_time = (end_time - start_time) / iterations  # 平均推理时间

print(f"Average inference time per batch: {inference_time:.4f} seconds")
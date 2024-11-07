import torch
from utils import measure_latency

# load your net
net = your_net()
images = torch.zeros(32,3,224,224)
avg_throughput = 0
avg_latency = 0

# average 5 times
for _ in range(5):
    throughput, latency = measure_latency(images,net, GPU=True, chan_last=False, half=True, num_threads=None, iter=500)
    avg_throughput += throughput
    avg_latency += latency

final_throughput = avg_throughput / 5
final_latency = avg_latency / 5
print(f"5 average throughput on gpu {final_throughput}")
print(f"5 average latency on gpu {final_latency} ms")
import torch
import time

x = torch.rand((4096, 4096), device='cuda')
for i in range(5000):
    start = time.time()
    y = x @ x
    torch.cuda.synchronize()
    print(f"Iteration {i}, time: {time.time() - start:.4f}")

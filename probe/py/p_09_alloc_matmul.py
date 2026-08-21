import torch
x = torch.ones((8, 8), dtype=torch.float16, device="cuda")
print("matmul sum =", (x @ x).sum().item())

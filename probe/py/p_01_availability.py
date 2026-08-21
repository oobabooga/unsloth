import torch
print("torch", torch.__version__, "hip", torch.version.hip)
print("is_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())

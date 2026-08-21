import torch
p = torch.cuda.get_device_properties(0)
print("total_memory  =", p.total_memory)
print("mem_get_info  =", torch.cuda.mem_get_info(0))
print("mem_reserved  =", torch.cuda.memory_reserved())
print("mem_allocated =", torch.cuda.memory_allocated())

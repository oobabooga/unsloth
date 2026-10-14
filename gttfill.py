"""Hold GTT through RADV so the free pool matches a target size: allocates and touches host-visible
Vulkan memory from the largest non-VRAM heap in 1 GiB chunks, reports sysfs GTT use, then sleeps."""
import ctypes, glob, sys, time
import vulkan as vk

def gtt():
    d = [p.rsplit("/", 1)[0] for p in glob.glob("/sys/class/drm/card*/device/mem_info_gtt_total")]
    tot = sum(int(open(x + "/mem_info_gtt_total").read()) for x in d)
    used = sum(int(open(x + "/mem_info_gtt_used").read()) for x in d)
    return tot, used

target_free = float(sys.argv[1]) * 1024**3
inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=vk.VkApplicationInfo(apiVersion=vk.VK_MAKE_VERSION(1, 2, 0))), None)
pdev = vk.vkEnumeratePhysicalDevices(inst)[0]
props = vk.vkGetPhysicalDeviceMemoryProperties(pdev)
want = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
types = []
for i in range(props.memoryTypeCount):
    t = props.memoryTypes[i]; h = props.memoryHeaps[t.heapIndex]
    print(f"type {i} flags=0x{t.propertyFlags:x} heap {t.heapIndex} size={h.size/2**30:.2f} GiB heapflags=0x{h.flags:x}", flush=True)
    if (t.propertyFlags & want) == want and not (t.propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT):
        types.append((h.size, i))
mtype = max(types)[1]
print("using memory type", mtype, flush=True)
qci = vk.VkDeviceQueueCreateInfo(queueFamilyIndex=0, queueCount=1, pQueuePriorities=[1.0])
dev = vk.vkCreateDevice(pdev, vk.VkDeviceCreateInfo(queueCreateInfoCount=1, pQueueCreateInfos=[qci]), None)
tot, used = gtt()
need = int(tot - used - target_free)
print(f"gtt total={tot/2**30:.2f} used={used/2**30:.2f} GiB; allocating {need/2**30:.2f} GiB", flush=True)
held = []
chunk = 1 << 30
while need > 0:
    n = min(chunk, need)
    mem = vk.vkAllocateMemory(dev, vk.VkMemoryAllocateInfo(allocationSize=n, memoryTypeIndex=mtype), None)
    ptr = vk.vkMapMemory(dev, mem, 0, n, 0)
    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(ptr)), 1, n)
    held.append(mem); need -= n
tot, used = gtt()
print(f"FILLED gtt used={used/2**30:.2f} free={(tot-used)/2**30:.2f} GiB", flush=True)
while True:
    time.sleep(60)

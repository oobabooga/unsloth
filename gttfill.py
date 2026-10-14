"""Hold amdgpu memory through RADV so free GTT (and optionally free VRAM) match targets, then sleep.

usage: gttfill.py GTT_FREE_GIB [VRAM_FREE_GIB]. Allocates 1 GiB chunks: host-visible non-device-local
memory (GTT) until sysfs free GTT reaches the target, and device-local memory until free VRAM does."""
import ctypes, glob, os, sys, time
import vulkan as vk

G = 1024**3

HEAPS = {}

def pool(kind):
    if os.name == "nt":
        # No sysfs: treat the memory heap as the pool and only this process as its user.
        return HEAPS.get(kind, 0), sum(1 for _ in held_kind(kind)) * CHUNK
    d = [p.rsplit("/", 1)[0] for p in glob.glob(f"/sys/class/drm/card*/device/mem_info_{kind}_total")]
    tot = sum(int(open(x + f"/mem_info_{kind}_total").read()) for x in d)
    used = sum(int(open(x + f"/mem_info_{kind}_used").read()) for x in d)
    return tot, used

inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=vk.VkApplicationInfo(apiVersion=vk.VK_MAKE_VERSION(1, 2, 0))), None)
pdev = vk.vkEnumeratePhysicalDevices(inst)[0]
props = vk.vkGetPhysicalDeviceMemoryProperties(pdev)
HV = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
DL = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
gtt_types, vram_types = [], []
for i in range(props.memoryTypeCount):
    t = props.memoryTypes[i]; h = props.memoryHeaps[t.heapIndex]
    print(f"type {i} flags=0x{t.propertyFlags:x} heap {t.heapIndex} size={h.size/G:.2f} GiB", flush=True)
    if t.propertyFlags & 0xC0:
        continue  # AMD device-coherent / uncached types need VK_AMD_device_coherent_memory enabled
    if (t.propertyFlags & HV) == HV and not (t.propertyFlags & DL):
        gtt_types.append((h.size, i))
    if (t.propertyFlags & DL) and not (t.propertyFlags & vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT):
        vram_types.append((h.size, i))
if gtt_types:
    HEAPS["gtt"] = max(gtt_types)[0]
if vram_types:
    HEAPS["vram"] = max(vram_types)[0]
qci = vk.VkDeviceQueueCreateInfo(queueFamilyIndex=0, queueCount=1, pQueuePriorities=[1.0])
dev = vk.vkCreateDevice(pdev, vk.VkDeviceCreateInfo(queueCreateInfoCount=1, pQueueCreateInfos=[qci]), None)
held = []
held_kinds = []

def held_kind(kind):
    return [k for k in held_kinds if k == kind]

CHUNK = 256 << 20


def fill(kind, mtype, target_free, touch):
    tot, used = pool(kind)
    need = int(tot - used - target_free * G)
    candidates = [mtype] + [i for _, i in sorted(gtt_types if kind == "gtt" else vram_types, reverse=True) if i != mtype]
    print(f"{kind} total={tot/G:.2f} used={used/G:.2f} GiB; allocating {max(need,0)/G:.2f} GiB, types {candidates}", flush=True)
    while need > 0:
        n = min(CHUNK, need)
        mem = None
        for t in list(candidates):
            try:
                mem = vk.vkAllocateMemory(dev, vk.VkMemoryAllocateInfo(allocationSize=n, memoryTypeIndex=t), None)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"type {t} refused {n} bytes: {type(exc).__name__}", flush=True)
                candidates.remove(t)
        if mem is None:
            print("no memory type accepts the allocation; stopping", flush=True)
            break
        if touch:
            ptr = vk.vkMapMemory(dev, mem, 0, n, 0)
            ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(ptr)), 1, n)
        held.append(mem); held_kinds.append(kind); need -= n
        if len(held) % 16 == 0:
            print(f"held {len(held) * CHUNK / G:.2f} GiB", flush=True)

if len(sys.argv) > 2 and vram_types:
    fill("vram", max(vram_types)[1], float(sys.argv[2]), False)
fill("gtt", max(gtt_types)[1], float(sys.argv[1]), True)
for kind in ("vram", "gtt"):
    tot, used = pool(kind)
    print(f"FILLED {kind} used={used/G:.2f} free={(tot-used)/G:.2f} GiB", flush=True)
while True:
    time.sleep(60)

# Logs shell-level change notifications (those raised by SHChangeNotify calls, not by the file
# system watcher) until a stop file appears or the deadline passes.
import ctypes, sys, time, os
from ctypes import wintypes

log_path, stop_path, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
roots = sys.argv[4:]
user32 = ctypes.WinDLL("user32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)

class WNDCLASSW(ctypes.Structure):
    _fields_ = [("style", wintypes.UINT), ("lpfnWndProc", WNDPROC), ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HANDLE),
                ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HANDLE),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR)]

class ENTRY(ctypes.Structure):
    _fields_ = [("pidl", ctypes.c_void_p), ("fRecursive", wintypes.BOOL)]

class MSG(ctypes.Structure):
    _fields_ = [("hwnd", wintypes.HWND), ("message", wintypes.UINT), ("wParam", wintypes.WPARAM),
                ("lParam", wintypes.LPARAM), ("time", wintypes.DWORD), ("pt", wintypes.POINT)]

user32.DefWindowProcW.restype = LRESULT
user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.CreateWindowExW.restype = wintypes.HWND
user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p]
user32.PeekMessageW.argtypes = [ctypes.POINTER(MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT, wintypes.UINT]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(MSG)]
user32.TranslateMessage.argtypes = [ctypes.POINTER(MSG)]
shell32.SHChangeNotifyRegister.restype = wintypes.ULONG
shell32.SHChangeNotifyRegister.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG, wintypes.UINT,
                                           ctypes.c_int, ctypes.POINTER(ENTRY)]
shell32.SHChangeNotification_Lock.restype = wintypes.HANDLE
shell32.SHChangeNotification_Lock.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                              ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)),
                                              ctypes.POINTER(wintypes.LONG)]
shell32.SHChangeNotification_Unlock.argtypes = [wintypes.HANDLE]
shell32.SHGetPathFromIDListW.argtypes = [ctypes.c_void_p, wintypes.LPWSTR]
shell32.ILCreateFromPathW.restype = ctypes.c_void_p
shell32.ILCreateFromPathW.argtypes = [wintypes.LPCWSTR]
shell32.SHGetSpecialFolderLocation.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]

out = open(log_path, "a", encoding="utf-8", buffering=1)
WM_NOTIFY_ME = 0x0401

def path_of(p):
    if not p:
        return ""
    buf = ctypes.create_unicode_buffer(32768)
    return buf.value if shell32.SHGetPathFromIDListW(p, buf) else "<non-fs pidl>"

@WNDPROC
def wndproc(hwnd, msg, wp, lp):
    if msg == WM_NOTIFY_ME:
        pidls = ctypes.POINTER(ctypes.c_void_p)()
        ev = wintypes.LONG()
        lock = shell32.SHChangeNotification_Lock(wintypes.HANDLE(wp), wintypes.DWORD(lp & 0xFFFFFFFF),
                                                 ctypes.byref(pidls), ctypes.byref(ev))
        if lock:
            p1 = path_of(pidls[0]) if pidls else ""
            shell32.SHChangeNotification_Unlock(lock)
            out.write("%.3f\tEVENT\t0x%08x\t%s\n" % (time.time(), ev.value & 0xFFFFFFFF, p1))
        else:
            out.write("%.3f\tLOCKFAIL\t%d\n" % (time.time(), ctypes.get_last_error()))
        return 0
    return user32.DefWindowProcW(hwnd, msg, wp, lp)

wc = WNDCLASSW()
wc.lpfnWndProc = wndproc
wc.hInstance = kernel32.GetModuleHandleW(None)
wc.lpszClassName = "Pr11116Listener"
if not user32.RegisterClassW(ctypes.byref(wc)):
    out.write("REGISTERCLASS-FAILED %d\n" % ctypes.get_last_error()); sys.exit(1)
hwnd = user32.CreateWindowExW(0, "Pr11116Listener", "l", 0, 0, 0, 0, 0, None, None, wc.hInstance, None)
if not hwnd:
    out.write("CREATEWINDOW-FAILED %d\n" % ctypes.get_last_error()); sys.exit(1)

pidls = []
desk = ctypes.c_void_p()
shell32.SHGetSpecialFolderLocation(None, 0, ctypes.byref(desk))
pidls.append(desk.value)
for r in roots:
    pidls.append(shell32.ILCreateFromPathW(r))
entries = (ENTRY * len(pidls))(*[ENTRY(p, True) for p in pidls])
SHCNRF_ShellLevel, SHCNRF_NewDelivery = 0x0002, 0x8000
events = 0x00002000 | 0x08000000 | 0x00001000  # UPDATEITEM | ASSOCCHANGED | UPDATEDIR
reg = shell32.SHChangeNotifyRegister(hwnd, SHCNRF_ShellLevel | SHCNRF_NewDelivery, events,
                                     WM_NOTIFY_ME, len(pidls), entries)
out.write("REGISTER\t%d\troots=%d\n" % (reg, len(pidls)))
out.write("READY\n")
deadline = time.time() + seconds
msg = MSG()
while time.time() < deadline and not os.path.exists(stop_path):
    while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
        user32.TranslateMessage(ctypes.byref(msg))
        user32.DispatchMessageW(ctypes.byref(msg))
    time.sleep(0.01)
out.write("DONE\n")

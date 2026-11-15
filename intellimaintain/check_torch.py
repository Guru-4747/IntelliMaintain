import ctypes, glob, os, sys

lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
os.add_dll_directory(lib)

print("Python:", sys.executable)
print("\n--- System runtime DLLs ---")
for name in ["vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll", "libomp140.x86_64.dll"]:
    try:
        ctypes.WinDLL(name); print("OK     ", name)
    except OSError:
        print("MISSING", name)

print("\n--- torch\\lib DLLs ---")
for f in sorted(glob.glob(os.path.join(lib, "*.dll"))):
    try:
        ctypes.WinDLL(f); print("OK     ", os.path.basename(f))
    except OSError as e:
        print("FAIL   ", os.path.basename(f), "->", e)
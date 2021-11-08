import DeviceConfig;

if DeviceConfig.EnableGPU:
    import cupy as np;
    import cupyx as cpx;

    np.cuda.set_allocator(np.cuda.MemoryPool().malloc);

    print("\033[92m GPU Mode is Enabled (cupy) \033[0m");
else:
    import numpy as np;

    print("\033[91m GPU Mode is Disabled (numpy) \033[0m");
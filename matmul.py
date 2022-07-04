#!usr/bin/env python3
import numpy as np
import time
N = 2048;
if __name__ == "__main__":
    ts = time.monotonic();
    A = np.random.randn(N, N).astype(np.float32);
    B = np.random.randn(N, N).astype(np.float32);
    C =  A @ B;
    te = time.monotonic();

    flops = 2*N*N*N / (te - ts) * 1e-9;
    print(f'{flops} GFLOPS');
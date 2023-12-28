#!usr/bin/env python3
import numpy as np
import time
import os
os.environ['OMP_NUM_THREADS'] = '1'

N = 1024
def GetDataFromFile(filename):
    with open(filename, 'r') as f:
        l = [[np.float32(num) for num in line.split(' ')] for line in f]
        return l  
       
if __name__ == "__main__":
    
    A = np.random.randn(N, N).astype(np.float32);
    B = np.random.randn(N, N).astype(np.float32);
    
    # A = np.matrix(GetDataFromFile('matrix.dat')) 
    # B = np.matrix(GetDataFromFile('matrix.dat')) 
    # N = len(A)
    for i in range(15):           
        ts = time.monotonic();
        C =  A @ B
        te = time.monotonic();  

        flops = 2*N*N*N / (te - ts) * 1e-9;
        print(f'{flops} GFLOPS');

    with open("matmul", "wb") as f:
        f.write(A.data);
        f.write(B.data);
        f.write(C.data);

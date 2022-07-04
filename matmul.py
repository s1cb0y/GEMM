#!usr/bin/env python3
import numpy as np
import time

def GetDataFromFile(filename):
    with open(filename, 'r') as f:
        l = [[np.float32(num) for num in line.split(' ')] for line in f]
        return l  
       
if __name__ == "__main__":
    
    # A = np.random.randn(N, N).astype(np.float32);
    # B = np.random.randn(N, N).astype(np.float32);
  
    A = np.matrix(GetDataFromFile('matrix.dat')) 
    B = np.matrix(GetDataFromFile('matrix.dat')) 
    N = len(A)
   
    ts = time.monotonic();
    C =  A @ B;
    te = time.monotonic();   

    flops = 2*N*N*N / (te - ts) * 1e-9;
    print(f'{flops} GFLOPS');

GEMM written in C and optimized.
Multiple 10x faster than naive approach of matrix multiplication.
Multi threading could be used if compiled with -DTHREADS=2.
Compiles under linux, mac and windows

Please generate matrix data first by executing matmul.py (uses numpy)

clang -ffast-math -march=native -O3 matmul.c -DTHREADS=6 && ./a.out

Thanks geohot for inspiration and great youtube content!

TODO: Implement OpenCL matmul

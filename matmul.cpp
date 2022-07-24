#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <cassert>
#include <immintrin.h>
#define MATRIX_DATA_FILE "matrix.dat"

#define N 1024
#define BLOCK 8


float A[N*N];
float B[N*N];
float C[N*N];
float val[N*N];

__m256* Am = (__m256*) A;
__m256* Bm = (__m256*) B;
__m256* Cm = (__m256*) C;

uint64_t nanos(){
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &time);           
    return (uint64_t) time.tv_sec * 1e9 + (uint64_t) time.tv_nsec;
}


#ifndef FAST
void multiplyBlocked(){
    assert(N%BLOCK == 0);
    for (int rb = 0; rb < N; rb+=BLOCK){
        for (int cb = 0; cb < N; cb+=BLOCK){
            //compute
            float tb[BLOCK][BLOCK];
            for (int r = 0; r < BLOCK; r++){
                for (int c = 0; c < BLOCK; c++){
                    float acc = 0;
                    for (int k = 0; k < N; k++){
                        acc += A[(r+rb) * N +k] * B[(c+cb) * N + k];
                    }
                    C[(rb+r) * N + c+cb] = acc;                    
                }
            }
        }
    }    
}
#else
void multiplyBlocked(){
    assert(N%BLOCK == 0);
    for (int rb = 0; rb < N; rb+=BLOCK){
        for (int cb = 0; cb < N; cb+=BLOCK){
            //compute
            float tb[BLOCK][BLOCK];
            for (int r = 0; r < BLOCK; r++){       
                for (int c = 0; c < BLOCK; c++){                
                    __m256 acc = {};
                    for (int k = 0; k < N; k+=8){
                        acc = _mm256_fmadd_ps(Am[((r+rb) * N +k) / 8], Bm[((c+cb) * N + k) / 8], acc);
                    }
                    float facc = 0.0;
                    for (int i = 0; i < BLOCK; i++) facc += acc[i];
                    tb[r][c] = facc;      
                }              
            }
            // store
            for (int r = 0 ; r < BLOCK ; r++){
                for (int c = 0 ; c < BLOCK ; c++){
                    C[(rb+r) * N + c+cb] = tb[r][c];
                }
            }
        }    
    }
}
#endif
int main(){

    FILE *f = fopen("matmul", "rb");
    if (f){
        fread(A, sizeof(float), N*N, f);
        fread(B, sizeof(float), N*N, f);
        fread(val, sizeof(float), N*N, f);
        fclose(f);

        uint64_t start = nanos();
        multiplyBlocked();       
        uint64_t end = nanos();
        double flop = N*N*2.0*N;
        double s = (end - start) * 1e-9;
        std::cout << "GFlops:" << flop*1e-9 / s << std::endl;
        // validate against numpy
        for (int x = 0; x < N * N; x ++){  
            if (fabsf(C[x] - val[x]) > 1e-3){
                std::cout << "Matrix not equal at position: " << x << std::endl;
                std::cout << "C: " << C[x] << ", val:  " << val[x] << std::endl;
                return 1;
            }
        }      
    } else {
        std::cout << "File not found!\n";
        return 1;
    }


    return 0;    
}

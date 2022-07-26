//clang -ffast-math -march=native -O3 matmul.c && ./a.out
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>

#define N 1024
#define BLOCK_R 4
#define BLOCK_C 2


float A[N*N]; __attribute__ ((__aligned__((32))))
float B[N*N]; __attribute__ ((__aligned__((32))))
float C[N*N]; __attribute__ ((__aligned__((32))))
float val[N*N]; __attribute__ ((__aligned__((32))))

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
    assert(N%BLOCK_R == 0);
    assert(N%BLOCK_C == 0);
    for (int rb = 0; rb < N; rb+=BLOCK_R){
        for (int cb = 0; cb < N; cb+=BLOCK_C){
            //compute           
            for (int k = 0; k < N; k++){
                for (int r = 0; r < BLOCK_R; r++){
                    for (int c = 0; c < BLOCK_C; c++){
                            C[(rb+r) * N + c+cb] += A[(r+rb) * N +k] * B[(c+cb) * N + k];
                        }
                }
            }
        }
    }    
}
#else
void multiplyBlocked(){
    assert(N%BLOCK_R == 0);
    assert(N%BLOCK_C == 0);
    for (int rb = 0; rb < N; rb+=BLOCK_R){
        for (int cb = 0; cb < N; cb+=BLOCK_C){
            //compute
            float tb[BLOCK_R][BLOCK_C];
            for (int r = 0; r < BLOCK_R; r++){       
                for (int c = 0; c < BLOCK_C; c++){                
                    __m256 acc = {};
                    for (int k = 0; k < N; k+=8){
                        acc = _mm256_fmadd_ps(Am[((r+rb) * N +k) / 8], Bm[((c+cb) * N + k) / 8], acc);
                    }
                    float facc = 0.0;
                    for (int i = 0; i < 8; i++) facc += acc[i];
                    tb[r][c] = facc;      
                }              
            }
            // store
            for (int r = 0 ; r < BLOCK_R ; r++){
                for (int c = 0 ; c < BLOCK_C ; c++){
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
        printf ("GFlops: %f\n", flop*1e-9 / s);;
        // validate against numpy
        for (int x = 0; x < N * N; x ++){  
            if (fabsf(C[x] - val[x]) > 1e-3){
                printf("Matrix not equal at position: %d\n", x);
                printf("C: %f , val: %f ", C[x] , val[x]);
                return 1;
            }
        }      
    } else {
        printf( "File not found!\n");
        return 1;
    }


    return 0;    
}

//clang -ffast-math -march=native -O3 matmul.c && ./a.out
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include "matmulGPU.h"

#define N 1024
#define BLOCK_R 4
#define BLOCK_C 2


float A[N*N]; __attribute__ ((__aligned__((64))))
float B[N*N]; __attribute__ ((__aligned__((64))))
float C[N*N]; __attribute__ ((__aligned__((64))))
float val[N*N]; __attribute__ ((__aligned__((64))))


__m256* Am = (__m256*) A;
__m256* Bm = (__m256*) B;
__m256* Cm = (__m256*) C;

uint64_t nanos(){
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &time);           
    return (uint64_t) time.tv_sec * 1e9 + (uint64_t) time.tv_nsec;
}

void simpleMultiply(){
    for (int r=0; r<N; r++) {
        for (int c=0; c<N; c++) {
            float acc = 0.0f;
            for (int k=0; k<N; k++) {
                acc += A[r*N + k] * B[k*N + c];
            }
            C[r*N + c] = acc;
        }
    }
}

// void simpleMultiplyFast(){
    
//     for (int r=0; r<N; r++) {
//         for (int k=0; k<N; k++) {
//             for (int c=0; c<N; c++) {                
//                 C[r*N + c] += A[r*N + k] * B[k*N + c];
//             }
//         }
//     }
// }

const uint32_t TILE_SIZE = 4;
#define BLOCK_Y 8
#define BLOCK_X 2
#define BLOCK_F 8
// with tiling and
void simpleMultiplyFast(){
    
    for (int t = 0; t < N; t+=TILE_SIZE){
        for (int r=0; r<N; r++) {
            for (int k=t; k<t+TILE_SIZE; k++) {
                for (int c=0; c<N; c+=BLOCK_Y*BLOCK_F) {      
                    __m256 acc[BLOCK_Y] = {};                 
                    __m256 ar = _mm256_broadcast_ss(&A[(r)*N + k]);
                    for (int cb = 0; cb < BLOCK_Y; cb++){ // Force to use all 16 YMM registers
                        //acc[cb] += ar * B[k*N + c + cb];
                        acc[cb] += _mm256_fmadd_ps(ar, Bm[((k*N + c + cb*BLOCK_F)/8)], acc[cb]);
                    }                      
                
                    for (int cb = 0; cb < BLOCK_Y; cb++){ // store registers back to RAM
                        Cm[((r*N + c + cb*8) / 8)] += acc[cb];
                    }
                    
                }                
            }
        }
    }
}

// #ifndef FAST
// void multiplyBlocked(){
//     assert(N%BLOCK_R == 0);
//     assert(N%BLOCK_C == 0);
//     for (int rb = 0; rb < N; rb+=BLOCK_R){
//         for (int cb = 0; cb < N; cb+=BLOCK_C){
//            float tc[BLOCK_R][BLOCK_C] = {};
//             for (int k = 0; k < N; k++) {
//                 for (int y = 0; y < BLOCK_R; y++) {
//                     for (int x = 0; x < BLOCK_C; x++) {
//                         tc[y][x] += A[(rb+y)*N + k] * B[(cb+x)*N + k];
//                     }
//                 }
//             }
//             // store
//             for (int y = 0; y < BLOCK_R; y++) {
//                 for (int x = 0; x < BLOCK_C; x++) {
//                 C[(rb+y)*N + cb+x] = tc[y][x];
//                 }
//             }
//         }    
//     }
// }
//#else
void multiplyBlocked(){
    assert(N%BLOCK_R == 0);
    assert(N%BLOCK_C == 0);
    for (int rb = 0; rb < N; rb+=BLOCK_R){
        for (int cb = 0; cb < N; cb+=BLOCK_C){
             __m256 tc[BLOCK_R][BLOCK_C] = {};
            for (int k = 0; k < N; k += 8) {
                for (int y = 0; y < BLOCK_R; y++) {
                for (int x = 0; x < BLOCK_C; x++) {
                    //printf("%d %d\n", ((rb+y)*N + k)/8, ((cb+x)*N + k)/8);
                    tc[y][x] = _mm256_fmadd_ps(
                    Am[((rb+y)*N + k)/8],
                    Bm[((cb+x)*N + k)/8],
                    tc[y][x]);
                }
                }
            }

            // store
            for (int y = 0; y < BLOCK_R; y++) {
                for (int x = 0; x < BLOCK_C; x++) {
                float ftmp = 0.0;
                for (int i = 0; i < 8; i++) ftmp += tc[y][x][i];
                C[(rb+y)*N + cb+x] = ftmp;
                }
      }
        }    
    }
}
//#endif



int main(){

    FILE *f = fopen("matmul", "rb");
    if (f){
        fread(A, sizeof(float), N*N, f);
        fread(B, sizeof(float), N*N, f);
        fread(val, sizeof(float), N*N, f);
        fclose(f);
        uint64_t start;
        uint64_t end;
        double s;
        double flop = N*N*2.0*N;

        for (int i = 0; i < 4; i++) {
            // memset(C, 0, N*N*sizeof(float));
            // // simple multiply
            // start = nanos();
            // simpleMultiply();
            // end = nanos();
            
            // s = (end - start) * 1e-9;
            // printf ("GFlops (naive approach): %f\n", flop*1e-9 / s);
            
            memset(C, 0, N*N*sizeof(float));
            //simple multiply (optimized)
            start = nanos();
            simpleMultiplyFast();
            end = nanos();
            s = (end - start) * 1e-9;
            printf ("GFlops (naive optimized approach) / Time (s): %f, %f\n", flop*1e-9 / s, s);
            
            // //blocked CPU multiply
            // start = nanos();            
            // multiplyBlocked();             
            // end = nanos();
            // s = (end - start) * 1e-9;
            // printf ("GFlops (blocked approach): %f\n", flop*1e-9 / s);
            
            // memset(C, 0, N*N*sizeof(float));
            // // GPU multiply
            // start = nanos();
            // multiplyGPU(A, B, C, N);
            // end = nanos();
            // s = (end - start) * 1e-9;
            // printf ("GFlops (GPU approach): %f\n", flop*1e-9 / s);
        }
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

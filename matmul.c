//clang -ffast-math -march=native -O3 matmul.c && ./a.out
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
//#include "matmulGPU.h"

#define N 768
#define BLOCK_Y 32  
#define BLOCK_X 32 
#define BLOCK 8


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
const uint32_t TILE_SIZE = 4;

// re-arranged k-loop
// use tiling 
void simpleMultiplyFast(){
    for (int t=0; t<N; t+=TILE_SIZE){
      for (int r=0; r<N; r++) {
         for (int k=t; k<t+TILE_SIZE; k++) {
               for (int c=0; c<N; c++) {                
                  C[r*N + c] += A[r*N + k] * B[k*N + c];
               }
         }
      }
   }
}


// #ifndef FAST
void multiplyBlocked(){
      assert(N%BLOCK_X == 0);
      assert(N%BLOCK_Y == 0);
      for (int cb = 0; cb < N; cb+=BLOCK_X){
         for (int r = 0; r < N; r++){
            for (int k=0; k<N; k+=BLOCK_X){
               for(int rr=0; rr<BLOCK_Y; rr++){
                  for(int cc=0; cc<BLOCK_X; cc++){
                     C[r*N+cb+cc] += A[r*N+k+rr] * B[k*N+rr*N +cb + cc];
                  }
               }
            }
            }
      }
}
//#else
// void multiplyBlocked(){
//     assert(N%BLOCK_X == 0);
//     assert(N%BLOCK_Y == 0);
//     for (int rb = 0; rb < N; rb+=BLOCK_Y){
//         for (int cb = 0; cb < N; cb+=BLOCK_X){
//              __m256 tc[BLOCK_Y][BLOCK_X] = {};
//             for (int k = 0; k < N; k += 8) {
//                 for (int y = 0; y < BLOCK_Y; y++) {
//                 for (int x = 0; x < BLOCK_X; x++) {
//                     tc[y][x] = _mm256_fmadd_ps(
//                     Am[((rb+y)*N + k)/8],
//                     Bm[((cb+x)*N + k)/8],
//                     tc[y][x]);
//                 }
//                 }
//             }

//             // store
//             for (int y = 0; y < BLOCK_Y; y++) {
//                 for (int x = 0; x < BLOCK_X; x++) {
//                 float ftmp = 0.0;
//                 for (int i = 0; i < 8; i++) ftmp += tc[y][x][i];
//                 C[(rb+y)*N + cb+x] = ftmp;
//                 }
//       }
//         }    
//     }
// }
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
            
            // memset(C, 0, N*N*sizeof(float));
            // //blocked CPU multiply
            // start = nanos();            
            // multiplyBlocked();             
            // end = nanos();
            // s = (end - start) * 1e-9;
            // printf ("GFlops (blocked approach) / TIME (s): %f, %f \n", flop*1e-9 / s, s);
            
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

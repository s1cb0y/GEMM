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
#define BLOCK 8
#define BLOCK_Y 2  
#define BLOCK_X 4


float A[N*N]; __attribute__ ((__aligned__((64))))
float B[N*N]; __attribute__ ((__aligned__((64))))
float BT[N*N]; __attribute__ ((__aligned__((64))))
float C[N*N]; __attribute__ ((__aligned__((64))))
float val[N*N]; __attribute__ ((__aligned__((64))))

uint64_t nanos(){
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &time);           
    return (uint64_t) time.tv_sec * 1e9 + (uint64_t) time.tv_nsec;
}

void Transpose(){
    for (int y = 0; y<N; y++){
        for (int x = 0; x<N; x++){
            //printf("BT[%d], B[%d]\n", y*N + x, x*N+y);
            BT[y*N + x] = B[x*N+y];
        }
    }
}
//using transose B
void simpleMultiply(){
    for (int r=0; r<N; r++) {
        for (int c=0; c<N; c++) {
            float acc = 0.0f;
            for (int k=0; k<N; k++) {
                acc += A[r*N + k] * BT[c*N + k];
            }
            C[r*N + c] = acc;
        }
    }
}


void blockedMultiply(){
    for (int r=0; r<N; r+=BLOCK_Y) {
        for (int c=0; c<N; c+=BLOCK_X) {
            float acc[BLOCK_Y][BLOCK_X] = {};
            for (int k=0; k<N; k++) {
                //printf("**** k: %d***** \n", k);
                for (int rr=0; rr<BLOCK_Y; rr++){
                    float a = A[(r+rr)*N + k];
                    for (int cc=0; cc<BLOCK_X; cc++){       
                  //      printf("acc[%d,%d], A[%d,%d],  B[%d,%d]\n", rr, cc, (r+rr), k, (c+cc), k); 
                        acc[rr][cc] += a * BT[(c+cc)*N + k];
                    }
                }
            }
            for (int rr=0; rr<BLOCK_Y; rr++){
                for (int cc=0; cc<BLOCK_X; cc++){        
                    C[(r+rr)*N + c+cc] = acc[rr][cc];
                }
            }
        }
    }
}



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
        Transpose();

        for (int i = 0; i < 4; i++) {
            memset(C, 0, N*N*sizeof(float));
            // simple multiply
            // start = nanos();
            // simpleMultiply();
            // end = nanos();
            
            // s = (end - start) * 1e-9;
            // printf ("GFlops (naive approach): %f\n", flop*1e-9 / s);
            
            memset(C, 0, N*N*sizeof(float));
            //simple multiply (optimized)
            start = nanos();
            blockedMultiply();
            end = nanos();
            s = (end - start) * 1e-9;
            printf ("GFlops (blocked approach) / Time (s): %f, %f\n", flop*1e-9 / s, s);
           
    
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

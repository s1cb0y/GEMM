#include <fstream>
#include <iostream>
#include <vector>
#include <math.h>
#include <cassert>
#define MATRIX_DATA_FILE "matrix.dat"

#define N 4096
#define BLOCK 32

float A[N][N];
float B[N][N];
float C[N][N];
float val[N][N];

uint64_t nanos(){
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC_RAW, &time);           
    return (uint64_t) time.tv_sec * 1e9 + (uint64_t) time.tv_nsec;
}

void multiply(){
    for (int r = 0; r < N; r++){
        for (int c = 0; c < N; c++){
            float acc = 0;
            for (int k = 0; k < N; k++){
                acc += A[r][k] * B[k][c];
            }
            C[r][c] = acc;                    
        }
    }    
}

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
                        acc += A[r+rb][k] * B[k][c+cb];
                    }
                    tb[r][c] = acc;                    
                }
            }
            // store
            for (int r = 0 ; r < BLOCK ; r++){
                for (int c = 0 ; c < BLOCK ; c++){
                    C[rb+r][c+cb] = tb[r][c];
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

        uint64_t start = nanos();
        multiplyBlocked();       
        uint64_t end = nanos();
        double flop = N*N*2.0*N;
        double s = (end - start) * 1e-9;
        std::cout << "GFlops:" << flop*1e-9 / s << std::endl;
        // validate against numpy
        for (int x = 0; x < N; x ++){   
            for (int y = 0; y < N; y ++){
                if (fabsf(C[x][y] - val[x][y]) > 1e-3){
                    std::cout << "Matrix not equal at position: " << x << ", " << y << std::endl;
                    std::cout << "C: " << C[x][y] << ", val:  " << val[x][y] << std::endl;
                    return 1;
                }
            }
        }      
    } else {
        std::cout << "File not found!\n";
        return 1;
    }


    return 0;    
}

// not used
class FileParser{
    public:
    
    bool ReadDataFromFile(const std::string& filename){
        std::ifstream fstream {filename};
        if (!fstream){
            std::cerr << "Error, could not open file:" <<  MATRIX_DATA_FILE << std::endl;
            return false;
        }
        float data;
          
        while (fstream >> data){            
            matData.push_back(data);
        }
        if (fstream.eof()){
            return true;
        }
        if (fstream.bad()){
            std::cerr << "Stream is bad, read failed!" << std::endl;
            return false;
        }
        if (fstream.fail()){
            fstream.clear();
            double c;
            fstream >> c;
            std::cout << "Last element: " << c << std::endl;        
            return false;
        }
        return true;
    }

    void GetData(){
        int k = 0;
        for (int i = 0; i < N; i++){
            for (int j = 0; j < N; j++){                
                A[i][j] = matData[k]; 
                B[i][j] = matData[k];
                k++; 
            }   
        } 
    }

    private:
    std::vector<float> matData;
    
};
#include <fstream>
#include <iostream>
#include <vector>
#define MATRIX_DATA_FILE "matrix.dat"

#define N 512

float A[N][N];
float B[N][N];
float C[N][N];


class FileParser{
    public:
    
    bool ReadDataFromFile(const std::string& filename){
        std::ifstream fstream {filename};
        if (!fstream){
            std::cerr << "Error, could not open file:" <<  MATRIX_DATA_FILE << std::endl;
            return false;
        }
        float data;
        char comma;  
          
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
                A[i][j] = matData[k++]; 
                B[i][j] = matData[k++]; 
            }   
        } 
    }

    private:
    std::vector<float> matData;
    
};

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

int main(){

    FileParser fp = FileParser();
    if (fp.ReadDataFromFile(MATRIX_DATA_FILE)){
        fp.GetData();
        uint64_t start = nanos();
        multiply();       
        uint64_t end = nanos();
        double flop = N*N*2.0*N;
        double s = (end - start) * 1e-9;
        std::cout << "GFlops:" << flop*1e-9 / s << std::endl;
    } else {
        std::cerr << "something went wrong while parsing file!" << std::endl;
        return 1;
    }

    return 0;    
}
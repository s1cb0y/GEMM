#include <fstream>
#include <iostream>
#include <vector>
#define MATRIX_DATA_FILE "matrix.csv"

#define N 4
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

    std::vector<float>& GetData(){
        return matData;
    }

    private:
    std::vector<float> matData;
    
};
int main(){

    FileParser fp = FileParser();
    if (fp.ReadDataFromFile(MATRIX_DATA_FILE)){
        
    } else {
        std::cerr << "something went wrong while parsing file!" << std::endl;
        return 1;
    }

    return 0;    
}
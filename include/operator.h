//
// Created by zhou on 2019/9/17.
//

#ifndef LSTM_OPERATOR_H
#define LSTM_OPERATOR_H

#include <memory.h>
#include <cassert>
#include <malloc.h>
#include <cmath>
#include <data_type.h>
#include <fstream>
#include <cmath>

typedef unsigned int uint;
#define CIRCULANT_SIZE 8
// matrix multiplication
template<class Dtype>
Dtype *MatMul(Dtype *A, Dtype *B, uint A_row, uint A_col, uint B_row, uint B_col, bool is_circulant) {

    assert(A_col == B_row);
    uint C_row = A_row;
    uint C_col = B_col;

    //the result matrix
    Dtype *C = (Dtype *) malloc(C_row * C_col * sizeof(Dtype));

    for (int i = 0; i < A_row; i++) {
        for (int j = 0; j < B_col; j++) {
            Dtype sum = 0;
            for (int k = 0; k < A_col; k++) {
                int offset = i * A_col + k;
                if(is_circulant){
                    int block_i = i/CIRCULANT_SIZE;
                    int block_j = k/CIRCULANT_SIZE;
                    int pix_i = i%CIRCULANT_SIZE;
                    int pix_j = k%CIRCULANT_SIZE;
                    offset = block_i*A_col+block_j*CIRCULANT_SIZE+(pix_j+CIRCULANT_SIZE-pix_i)%CIRCULANT_SIZE;
                }

                sum += A[offset] * B[k * B_col + j];
            }
            C[i * C_col + j] = sum;
        }
    }
    return C;
}

//element wise multiplication
template<class Dtype>
Dtype *MatDot(Dtype *A, Dtype *B, uint row, uint col) {
    Dtype *C = (Dtype *) malloc(row * col * sizeof(Dtype));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            C[i * col + j] = A[i * col + j] * B[i * col + j];
        }
    }

    return C;
}

//Matrix addition (A+B+C)
template<class Dtype>
Dtype *MatAddTri(Dtype *A, Dtype *B, Dtype *C, uint row, uint col) {

    Dtype *out = (Dtype *) malloc(row * col * sizeof(Dtype));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            out[i * col + j] = A[i * col + j] + B[i * col + j] + C[i * col + j];
        }
    }
    return out;
}

//Matrix addition (A+B)
template<class Dtype>
Dtype *MatAddDuo(Dtype *A, Dtype *B, uint row, uint col) {
    Dtype *out = (Dtype *) malloc(row * col * sizeof(Dtype));
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            out[i * col + j] = A[i * col + j] + B[i * col + j];
        }
    }
    return out;
}

template<class Dtype>
class Activation {
public:
    Activation()
    {

    }

    Activation(const std::string &table_paths) {
        std::string sigmoid_path = table_paths + std::string("/sigmoid_table.txt");
        read_table_from_file(sigmoid_path, TABLE_SIZE, sigmoid_table);
        std::string tanh_path = table_paths + std::string("/tanh_table.txt");
        read_table_from_file(tanh_path, TABLE_SIZE, tanh_table);
    }
    static const int TABLE_SIZE = 512;

    Dtype sigmoid_table[TABLE_SIZE];
    Dtype tanh_table[TABLE_SIZE];

    void read_table_from_file(std::string &data_file, uint size, Dtype *pdata) {
        std::ifstream infile;
        infile.open(data_file.data());
        assert(infile.is_open());
        std::string s;
        //read shape
        getline(infile, s); // dismiss the first line (the first line is the shape)

        for (int i = 0; i < size; i++) {
            infile >> pdata[i];
        }
    }

    void generate_sigmoid_table()
    {
        Fixed_point<9,3,6> x;
        for(int i = 0; i < TABLE_SIZE; i++ )
        {
            x.data = i;
            float float_x = float(x);
            float float_sigmoid_x = 1/(exp(-float_x)+1);
            Fixed_point<16,3,13> fix_sigmoid_x(float_sigmoid_x);
            std::cout<<fix_sigmoid_x<<" "<<std::endl;
        }
    }

    void generate_tanh_table()
    {
        Fixed_point<9,3,6> x;
        for(int i = 0; i < TABLE_SIZE; i++ )
        {
            x.data = i;
            float float_x = float(x);
            float float_tanh_x = (exp(float_x) - exp(-float_x))/(exp(-float_x)+ exp(float_x));
            Fixed_point<16,3,13> fix_tanh_x(float_tanh_x);
            std::cout<<fix_tanh_x<<" "<<std::endl;
        }
    }

    Dtype f_sigmoid( Dtype &x) {
        int index;
        // truncate the x, get the first 9bits
        index = (x.data >> 7)&0x1FF;
        return sigmoid_table[index];
    }

    Dtype f_tanh( Dtype &x) {
        int index;
        // truncate the x, get the first 9bits
        index = (x.data >> 7)&0x1FF;
        return tanh_table[index];
    }

    Dtype *sigmoid(Dtype *A, uint row, uint col) {
        Dtype * out = (Dtype *) malloc(row * col * sizeof(Dtype));
        for(int i = 0;i<row;i++)
        {
            for(int j = 0;j<col;j++)
            {
                out[i*col + j] = f_sigmoid(A[i*col + j]);
            }
        }
        return out;
    }

    Dtype *tanh(Dtype *A, uint row, uint col) {
        Dtype * out = (Dtype *) malloc(row * col * sizeof(Dtype));
        for(int i = 0;i<row;i++)
        {
            for(int j = 0;j<col;j++)
            {
                out[i*col + j] = f_tanh(A[i*col + j]);
            }
        }
        return out;
    }
};


#endif //LSTM_OPERATOR_H

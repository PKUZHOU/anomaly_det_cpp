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
    Dtype f_sigmoid( Dtype &x) {
        int index;
        // We want to mapping x to the index ranging from 0 to 511 (because we have 512 values)
        // The look up table is generated with x ranging from -3. to 3.,
        // So now the index = (x/3.)*256 + 256, or x * 85.333333 + 256, we use constant k to represent 256/3.
        // Note that k has 8 integer bits, rather than 3 integer bits in the x.

        Fixed_point<16, 8, 8> k("5555"); // the float number 256/3. is 0x5555 in fix<8,8> format

        index = x.data * k.data >> 21; // fix<3,13> multiplies fix<8,8>, so we right shift 21 bits to get
                                       // the integer part of the output. The output should be  at least 10 bits
                                       // to represent -512 to 512. I use integer here for convenience.
        index = index + 256;           // then we add 256.

        if(index < 0)  // make sure it is within 0 to 511
            index = 0;
        else if(index > 511)
            index = 511;

        return sigmoid_table[index];
    }

    Dtype f_tanh( Dtype &x) {
        int index;
        Fixed_point<16, 8, 8> k("5555"); // k = 85.333333
        index = x.data * k.data >> 21;
        index = index + 256;
        if(index < 0)
            index = 0;
        else if(index > 511)
            index = 511;

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

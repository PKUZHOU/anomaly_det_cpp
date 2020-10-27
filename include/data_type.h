//
// Created by zhou on 2019/12/6.
//

#ifndef LSTM_DATA_TYPE_H
#define LSTM_DATA_TYPE_H
#include <iomanip>
#include <iostream>
#include <algorithm>

#include <vector>

#define CIRCULANT_SIZE 8

template <int tot_bits = 16>
class INT{
public:
    int data = 0; // use int32 to hold the  data
    int TOTBits = tot_bits;
    int INTBits = 3;

    int saturate(int b)  // truncate the fixed_point data to the valid range.
    {
        if(b> ((1<<(TOTBits-1))-1))
            return ((1<<(TOTBits-1))-1);
        if(b<-(1<<(TOTBits-1)))
            return -(1<<(TOTBits-1));
        return b;
    }
    INT() {}

    INT(const int &b){
        data = saturate(int(b*(1<<(TOTBits-INTBits))));
    }

    INT(const float &b){
        data = saturate(int(b*(1<<(TOTBits-INTBits))));
    }

    INT& operator = (const int &b)
    {
        data = saturate(int(b*(1<<(TOTBits-INTBits))));
        return *this;
    }


    INT& operator = (const INT &b)
    {
        if(TOTBits != b.TOTBits){
            data = b.data>>(TOTBits + INTBits - b.INTBits);
            data = saturate(data);
        }
        else{
            assert(b.INTBits == INTBits);
            data = b.data;
        }
        return *this;
    }


    // INT(const char raw[2]) // init from hex data , eg. "1F" now only support 8bit
    // {
    //     for(int i = 0;i<2;i++)
    //     {
    //         if('0'<=raw[i] and raw[i]<='9')
    //         {
    //             data+=((raw[i]-'0')<<(4*(1-i)));
    //         }
    //         else
    //         {
    //             data+=((raw[i]-'A'+10)<<(4*(1-i)));
    //         }
    //     }
    //     data = data<<(32-TOTBits)>>(32-TOTBits); // the shift operation guarantees the two's complement format for negative values
    // }

    INT(const char raw[4]) // init from hex data 
    {
        for(int i = 0;i<4;i++)
        {
            if('0'<=raw[i] and raw[i]<='9')
            {
                data+=((raw[i]-'0')<<(4*(3-i)));
            }
            else
            {
                data+=((raw[i]-'A'+10)<<(4*(3-i)));
            }
        }
        data = data<<(32-TOTBits)>>(32-TOTBits); // the shift operation guarantees the two's complement format for negative values
    }


    INT(const INT& b) //init from other INT data
    {
        data = saturate(b.data>>(TOTBits + INTBits - b.INTBits));
    }

    INT operator * (const INT &b)
    {
        INT c;
//        c.TOTBits = TOTBits + b.TOTBits;
//        c.INTBits = INTBits + b.INTBits;

        c.TOTBits = TOTBits;
        c.INTBits = INTBits;

        int a_data = data;
        int b_data = b.data;
//        int c_data = a_data*b_data;
        int c_data = a_data * b_data >> (TOTBits-INTBits);
        c.data = c_data;
        return c;
    }

    INT operator + (const INT &b)
    {
        assert(b.TOTBits == TOTBits);
        assert(b.INTBits == INTBits);

        INT c;
        c.TOTBits = TOTBits;
        c.INTBits = INTBits;

        int a_data = data;
        int b_data = b.data;
        int c_data = saturate(a_data+b_data);

        c.data = c_data;
        return c;
    }

    INT operator - (const INT &b)
    {
        assert(b.TOTBits == TOTBits);
        assert(b.INTBits == INTBits);

        INT c;
        c.TOTBits = TOTBits;
        c.INTBits = INTBits;

        int a_data = data;
        int b_data = b.data;
        int c_data = saturate(a_data-b_data);

        c.data = c_data;
        return c;
    }

    INT operator += (const INT &b)
    {

        int b_data = b.data;
        int c_data = saturate(b_data + data);
        data = c_data;
        return *this;
    }

    // print the fixed point data in hex format
    friend std::ostream& operator <<(std::ostream& os, const INT& b) {
        int b_data = b.data;

        if(b.TOTBits == 16){
            b_data = b_data&0xFFFF;
            for(int i = 0;i<4;i++) {
                int tmp = ((b_data >> (4 * (3 - i))) & 0xF);
                if (tmp <= 9) {
                    os << char(tmp + '0');
                } else {
                    os << char(tmp - 10 + 'A');
                }
            }
        }
        else if(b.TOTBits == 8){
            b_data = b_data&0xFF;
            for(int i = 0;i<2;i++) {
                int tmp = ((b_data >> (4 * (1 - i))) & 0xF);
                if (tmp <= 9) {
                    os << char(tmp + '0');
                } else {
                    os << char(tmp - 10 + 'A');
                }
            }
        }
        return os;
    }

    // fill it using std::in
    // eg: cin>>b
    friend std::istream& operator >>(std::istream& is,  INT& b) {
        std::string tmp;
        is>>tmp;
        transform(tmp.begin(),tmp.end(),tmp.begin(),::toupper);
        auto new_data = INT(tmp.c_str()).data;
        b.data = new_data;
        return is;
    }
};

template <class Dtype>
class Mat{
public:
    uint row;
    uint col;
    uint intbits;
    Dtype * data;
    bool is_circulant = false;

    Mat(const int row_, const int col_, const uint intbits_){
        row = row_;
        col = col_;
        intbits = intbits_;
        data = (Dtype*) malloc(row * col * sizeof(Dtype));

        Dtype temp;

        // set the intbits
        for (int i = 0;i<row*col;i++){
            data[i].TOTBits = temp.TOTBits;
            data[i].INTBits = intbits_;
        }

        // set 0
        for (int i = 0;i<row*col;i++){
            data[i].data = 0;
        }
    }

    Mat(Dtype* data_, const int row_, const int col_, const uint intbits_){
        row = row_;
        col = col_;
        intbits = intbits_;
        data = (Dtype*) malloc(row * col * sizeof(Dtype));
        memcpy(data, data_, row * col * sizeof(Dtype));
        for (int i = 0;i<row*col;i++){
            data[i].INTBits = intbits_;
        }
    }

    Mat(const Mat & b){
        row = b.row;
        col = b.col;
        intbits = b.intbits;
        data = (Dtype*) malloc(row * col * sizeof(Dtype));
        memcpy(data, b.data, row * col * sizeof(Dtype));
        for (int i = 0;i<row*col;i++){
            data[i].INTBits = intbits;
        }
    }

    Dtype& operator[](int i){
        return *(data + i);
    }

    Mat<Dtype>* matmul(Mat* b, uint intbits){

        uint A_col = col;
        uint A_row = row;
        uint B_row = b->row;
        uint B_col = b->col;

        assert(A_col == B_row);
        uint C_row = row;
        uint C_col = b->col;

        Mat<Dtype>* C = new Mat<Dtype>(C_row, C_col, intbits);
        for (int i = 0; i < A_row; i++) {
            for (int j = 0; j < B_col; j++) {

                INT<> sum = 0;
//                sum.TOTBits = data[0].TOTBits + b->data[0].TOTBits;
                sum.TOTBits = data[0].TOTBits;
//                sum.INTBits = this->intbits + b->intbits;
                sum.INTBits = this->intbits;

                for (int k = 0; k < A_col; k++) {
                    int offset = i * A_col + k;
                    auto out = data[offset] * b->data[k * B_col + j];
                    sum += out;
                }
                C->data[i * C_col + j] = sum;
            }
        }
        return C;
    }

    Mat<Dtype>* circ_mul(Mat* b, uint intbits){

        uint A_col = col;
        uint A_row = row * CIRCULANT_SIZE;
        uint B_row = b->row;
        uint B_col = b->col;



        assert(A_col == B_row);
        uint C_row = A_row;
        uint C_col = b->col;

        Mat<Dtype>* C = new Mat<Dtype>(C_row, C_col, intbits);
        for (int i = 0; i < A_row; i++) {
            for (int j = 0; j < B_col; j++) {

                INT<> sum = 0;
//                sum.TOTBits = data[0].TOTBits + b->data[0].TOTBits;
                sum.TOTBits = data[0].TOTBits;
//                sum.INTBits = this->intbits + b->intbits;
                sum.INTBits = this->intbits;

                for (int k = 0; k < A_col; k++) {
//                    int offset = i * A_col + k;

                    int block_i = i/CIRCULANT_SIZE;
                    int block_j = k/CIRCULANT_SIZE;
                    int pix_i = i%CIRCULANT_SIZE;
                    int pix_j = k%CIRCULANT_SIZE;
                    int offset = block_i*A_col+block_j*CIRCULANT_SIZE+(pix_j+CIRCULANT_SIZE-pix_i)%CIRCULANT_SIZE;

                    auto out = data[offset] * b->data[k * B_col + j];
                    sum += out;
                }
                C->data[i * C_col + j] = sum;
            }
        }
        return C;
    }

    Mat<Dtype>* matadd(Mat *b){
        uint A_col = col;
        uint A_row = row;
        uint B_row = b->row;
        uint B_col = b->col;

        assert(A_row == B_row);
        assert(A_col == B_col);
        assert(this->intbits == b->intbits);

        Mat<Dtype>* c = new Mat<Dtype>(A_row, A_col, this->intbits);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                (*c)[i * col + j] = data[i * col + j] + (*b)[i * col + j];
            }
        }
        return c;
    }

    Mat<Dtype> * one_sub(){
        uint A_col = col;
        uint A_row = row;
        Mat<Dtype>* c = new Mat<Dtype>(A_row, A_col, this->intbits);

        Dtype one(1);
        one.INTBits = this->intbits;

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                (*c)[i * col + j] = one-data[i * col + j] ;
            }
        }
        return c;
    }


    Mat<Dtype>* mul(Mat *b, uint intbits){
        uint A_col = col;
        uint A_row = row;
        uint B_row = b->row;
        uint B_col = b->col;

        assert(A_row == B_row);
        assert(A_col == B_col);

        Mat<Dtype>* c = new Mat<Dtype>(A_row, A_col, intbits);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                (*c)[i * col + j] = data[i * col + j] * (*b)[i * col + j];
            }
        }
        return c;
    }

    Mat<Dtype>* add_scalar(Mat *b){
        uint A_col = col;
        uint A_row = row;

        Mat<Dtype>* c = new Mat<Dtype>(A_row, A_col, this->intbits);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < col; j++) {
                (*c)[i * col + j] = data[i * col + j] + (*b)[0];
            }
        }
        return c;
    }

};

#endif //LSTM_DATA_TYPE_H

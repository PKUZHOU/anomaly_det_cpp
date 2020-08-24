//
// Created by zhou on 2019/9/17.
//

#ifndef LSTM_FC_H
#define LSTM_FC_H

#include "operator.h"
#include <string>

template <class Dtype>
class FC{
    /*Fully connected layer  f(x) = w*x + b where w denotes weight matrix b is bias vector  */
public:
    uint input_size;     // equals to the columns number of weight matrix
    uint output_size;    // equals to the rows number of weight matrix

    Mat<Dtype> * weight;      // pointer to the weight
    Mat<Dtype> * bias;        // pointer to the bias
    Mat<Dtype> * out;  // pointer to the final output
    uint S_weight;
    uint S_bias;
    uint S_w_i;

public:
    FC(uint input_size, uint out_size)
    {
        /*construction function, initialize the input size and output size*/
        this->input_size = input_size;
        this->output_size = out_size;
    }
    void read_weights_from_file(std::string & weights_file, uint size, Mat<Dtype> * pweights)
    {
        std::ifstream infile;
        infile.open(weights_file.data());
        assert(infile.is_open());
        std::string s;
        //read shape
        getline(infile, s);

        for(int i = 0;i<size;i++)
        {
            infile >> (*pweights)[i];
        }
    }

    void read_scale_from_file(std::string & scale_file, uint & p_scale)
    {
        std::ifstream infile;
        infile.open(scale_file.data());
        assert(infile.is_open());
        std::string s;
//        getline(infile, s);
        infile >> p_scale;
    }

    void load_params(std::string& param_path)
    {
        cout<<"loading fc "<<endl;

        string weight_path = param_path + string("/weight");
        string scale_path = param_path + string("/scale");

        #define LOAD_SCALE(file_name, data)  \
                scale_file = scale_path+string("/")+string(file_name)+string(".txt"); \
                read_scale_from_file(scale_file,data);

        string scale_file;
        LOAD_SCALE("weight_fc", this->S_weight)
        LOAD_SCALE("bias_fc",this->S_bias)
        LOAD_SCALE("weight@i_fc",this->S_w_i)


        #define LOAD(data, row, col, scale, file_name) \
                data = new Mat<Dtype>(row,  col, scale); \
                weight_file = weight_path+ string("/") + string(file_name) +string(".txt"); \
                read_weights_from_file(weight_file, row * col, data);

        string weight_file;
        LOAD(this->weight,this->input_size,this->output_size,S_weight, "fc_w")
        LOAD( this->bias, 1, this->output_size, S_bias, "fc_bias")
    }

    void forward(Mat<Dtype> * x){
        Mat<Dtype> * fc_w_i = x->matmul(this->weight, S_w_i);
        out = fc_w_i->matadd(bias);
    }
};

#endif //LSTM_FC_H

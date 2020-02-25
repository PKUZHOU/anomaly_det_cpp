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
private:
    uint input_size;     // equals to the columns number of weight matrix
    uint output_size;    // equals to the rows number of weight matrix
    Dtype * weight;      // pointer to the weight
    Dtype * bias;        // pointer to the bias

public:
    Dtype * out;  // pointer to the final output

    FC(uint input_size, uint out_size)
    {
        /*construction function, initialize the input size and output size*/
        this->input_size = input_size;
        this->output_size = out_size;
    }
    void read_weights_from_file(std::string & weights_file, uint size, Dtype * pweights)
    {
        std::ifstream infile;
        infile.open(weights_file.data());
        assert(infile.is_open());
        std::string s;
        //read shape
        getline(infile, s);
//        cout<<s<<endl;
        for(int i = 0;i<size;i++)
        {
            infile >> pweights[i];
        }
    }
    void load_params(std::string& weight_path)
    {
        cout<<"loading fc "<<endl;

        this->weight = (Dtype*) malloc(this->input_size * this->output_size * sizeof(Dtype));
        this->bias = (Dtype*) malloc(this->output_size * sizeof(Dtype));

        string weight_file = weight_path+string("/fc_w.txt");
        read_weights_from_file(weight_file, this->input_size * this->output_size , this->weight);

        weight_file = weight_path+string("/fc_b.txt");
        read_weights_from_file(weight_file, this->output_size, this->bias);
    }

    void forward(Dtype * x){
        //w@x
        Dtype * w_mm_x =  MatMul(this->weight, x, output_size, input_size, input_size, 1,false);
        //w@x + b
        this->out = MatAddDuo(w_mm_x, bias, output_size, 1 );
        free(w_mm_x);
    }
};

#endif //LSTM_FC_H

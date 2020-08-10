//
// Created by zhou on 2019/9/17.
//

#ifndef LSTM_FC_H
#define LSTM_FC_H

#include "operator.h"
#include <string>

template <class Dtype, class T_temp>
class FC{
    /*Fully connected layer  f(x) = w*x + b where w denotes weight matrix b is bias vector  */
private:
    uint input_size;     // equals to the columns number of weight matrix
    uint output_size;    // equals to the rows number of weight matrix
    Dtype * weight;      // pointer to the weight
    Dtype * bias;        // pointer to the bias

    int w_scale;
    int b_scale;

public:
    Dtype * out;  // pointer to the final output

    FC(uint input_size, uint out_size)
    {
        /*construction function, initialize the input size and output size*/
        this->input_size = input_size;
        this->output_size = out_size;
    }
    void read_weights_from_file(std::string & weights_file, uint size, Dtype * pweights, int scale = 0)
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
            pweights[i].SCALE = scale;
        }
    }

    void read_scale_from_file(std::string & scale_file, int * scale)
    {
        std::ifstream infile;
        infile.open(scale_file.data());
        assert(infile.is_open());
        std::string s;
        getline(infile, s);
        infile >> *scale;
    }

    void load_params(std::string& weight_path)
    {
        cout<<"loading fc "<<endl;

        this->weight = (Dtype*) malloc(this->input_size * this->output_size * sizeof(Dtype));
        this->bias = (Dtype*) malloc(this->output_size * sizeof(Dtype));

        string w_scale_file = weight_path+string("/fc_w_scale.txt");
        read_scale_from_file(w_scale_file, &this->w_scale);

        string b_scale_file = weight_path+string("/fc_b_scale.txt");
        read_scale_from_file(b_scale_file,&this->b_scale);

        string weight_file = weight_path+string("/fc_w.txt");
        read_weights_from_file(weight_file, this->input_size * this->output_size , this->weight, this->w_scale);

        weight_file = weight_path+string("/fc_b.txt");
        read_weights_from_file(weight_file, this->output_size, this->bias, this->b_scale);
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

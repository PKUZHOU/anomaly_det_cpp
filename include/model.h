//
// Created by zhou on 2019/9/17.
//

#ifndef LSTM_MODEL_H
#define LSTM_MODEL_H

//#include "lstm.h"
#include "fast_grnn.h"
#include "fc.h"
#include <string>

template<class Dtype>
class FAST_GRNN {
private:
    fast_grnn_cell<Dtype> *fast_grnn_0 = NULL;  // first  layer
    fast_grnn_cell<Dtype> *fast_grnn_1 = NULL;  // second  layer
    FC<Dtype> * fc = NULL;                      // final fc layer

public:
    //hidden state and cell state
    //these states will be used and updated at every time step
    uint input_size;
    uint hidden_size;
    Mat<Dtype>* h_state = NULL;

    Mat<Dtype> *out;

    FAST_GRNN(uint input_size_, uint hidden_size_, uint out_size, std::string param_path) {
        hidden_size = hidden_size_;
        input_size = input_size_;

        this->fast_grnn_0 = new fast_grnn_cell<Dtype>(input_size, hidden_size, 0);

        //the second lstm layer takes the hidden_state of first lstm layer as input, so the input_size equals to hiden_size
        this->fast_grnn_1 = new fast_grnn_cell<Dtype>(input_size, hidden_size, 1);

        //fully connected layer takes the hidden_state of last lstm layer as input, so the input_size equals to hiden_size
        this->fc = new FC<Dtype>(hidden_size, out_size);
        //load weight params of all layers
        this->load_params(param_path);
    }

    ~FAST_GRNN() {
        delete(this->fast_grnn_0);
        delete(this->fast_grnn_1);
        delete(this->fc);
        free(this->out);
    };

    void reset() {
        //clear the hidden state and cell state
        if (!this->h_state) {
            this->h_state = new Mat<Dtype>(1, hidden_size,  this->fast_grnn_0->S_z_state);
        }
    }

    void forward(Dtype *x, uint time_steps) {
        /*Each model forward takes a sequence with length of 'time_steps' as input*/
        //At the beginning of each model forward, the hidden state and cell state of lstm layers should be reset.
        reset();

        uint S_in = this->fast_grnn_0->S_in;

        for (int st = 0; st < time_steps; st++) {
            //sequentially feed the inputs
            Mat<Dtype> x_st(&x[st],1,1,S_in);

            this->fast_grnn_0->forward(&x_st, h_state);
            this->fast_grnn_1->forward(&x_st, h_state);

            this->fc->forward(h_state);
            this->out = this->fc->out;
        }
    }

    void load_params(std::string param_path) {
        this->fast_grnn_0->load_params(param_path);
        this->fast_grnn_1->load_params(param_path);
        this->fc->load_params(param_path);
    }
};

#endif //LSTM_MODEL_H

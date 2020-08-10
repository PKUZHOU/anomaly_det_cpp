//
// Created by zhou on 2019/9/17.
//

#ifndef LSTM_MODEL_H
#define LSTM_MODEL_H

//#include "lstm.h"
#include "fast_grnn.h"
#include "fc.h"
#include <string>

template<class Dtype, class T_temp>
class FAST_GRNN {
private:
    fast_grnn_kernel<Dtype, T_temp> *fast_grnn_0 = NULL;  // first  layer
    fast_grnn_kernel<Dtype, T_temp> *fast_grnn_1 = NULL;  // second  layer
    FC<Dtype, T_temp> * fc = NULL;                      // final fc layer

public:
    Dtype *out;

    FAST_GRNN(uint input_size, uint hidden_size, uint out_size, std::string weights) {
        this->fast_grnn_0 = new fast_grnn_kernel<Dtype, T_temp>(input_size, hidden_size, 0);

        //the second lstm layer takes the hidden_state of first lstm layer as input, so the input_size equals to hiden_size
        this->fast_grnn_1 = new fast_grnn_kernel<Dtype, T_temp>(hidden_size, hidden_size, 1);

        //fully connected layer takes the hidden_state of last lstm layer as input, so the input_size equals to hiden_size
        this->fc = new FC<Dtype, T_temp>(hidden_size, out_size);
        //load weight params of all layers
        this->load_params(weights);
    }

    ~FAST_GRNN() {
        delete(this->fast_grnn_0);
        delete(this->fast_grnn_1);
        delete(this->fc);
        free(this->out);
    };

    void forward(Dtype *x, uint time_steps) {
        /*Each model forward takes a sequence with length of 'time_steps' as input*/
        //At the beginning of each model forward, the hidden state and cell state of lstm layers should be reset.
        this->fast_grnn_0->reset();
        this->fast_grnn_1->reset();

        for (int st = 0; st < time_steps; st++) {
            //sequentially feed the inputs
            this->fast_grnn_0->forward(&x[st]);
            this->fast_grnn_1->forward(this->fast_grnn_0->h_state);

            this->fc->forward(this->fast_grnn_1->h_state);
            this->out = this->fc->out;
        }
    }

    void load_params(std::string weights_path) {
        this->fast_grnn_0->load_params(weights_path);
        this->fast_grnn_1->load_params(weights_path);
        this->fc->load_params(weights_path);
    }
};

#endif //LSTM_MODEL_H

//
// Created by zhou on 2019/9/17.
//

#ifndef LSTM_MODEL_H
#define LSTM_MODEL_H

#include "lstm.h"
#include "fc.h"
#include <string>

template<class Dtype>
class Model {
private:
    lstm_kernel<Dtype> *lstm_0 = NULL;  // first lstm layer
    lstm_kernel<Dtype> *lstm_1 = NULL;  // second lstm layer
    FC<Dtype> *fc;                      // final fc layer

public:
    Dtype *out;

    Model(uint input_size, uint hidden_size, uint out_size, std::string weights) {
        this->lstm_0 = new lstm_kernel<Dtype>(input_size, hidden_size, 0);

        for(int idx = 4;idx<=7;idx++)
            this->lstm_0->is_circulant[idx] = 1;

        //the second lstm layer takes the hidden_state of first lstm layer as input, so the input_size equals to hiden_size
        this->lstm_1 = new lstm_kernel<Dtype>(hidden_size, hidden_size, 1);

        for(int idx = 0;idx<=7;idx++)
            this->lstm_1->is_circulant[idx] = 1;

        //fully connected layer takes the hidden_state of last lstm layer as input, so the input_size equals to hiden_size
        this->fc = new FC<Dtype>(hidden_size, out_size);
        //load weight params of all layers
        this->load_params(weights);
    }

    ~Model() {
        delete(this->lstm_0);
        delete(this->lstm_1);
        delete(this->fc);
        free(this->out);
    };

    void forward(Dtype *x, uint time_steps) {
        /*Each model forward takes a sequence with length of 'time_steps' as input*/
        //At the beginning of each model forward, the hidden state and cell state of lstm layers should be reset.
        this->lstm_0->reset();
        this->lstm_1->reset();

        for (int st = 0; st < time_steps; st++) {
            //sequentially feed the inputs
            this->lstm_0->forward(&x[st]);
            this->lstm_1->forward(this->lstm_0->h_state);

            this->fc->forward(this->lstm_1->h_state);
            this->out = this->fc->out;
        }
    }

    void load_params(std::string weights_path) {
        this->lstm_0->load_params(weights_path);
        this->lstm_1->load_params(weights_path);
        this->fc->load_params(weights_path);
    }
};


#endif //LSTM_MODEL_H

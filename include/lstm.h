//
// Created by zhou on 2019/9/17.
//

#ifndef LSTM_LSTM_H
#define LSTM_LSTM_H

#include "operator.h"
#include <string>
#include <fstream>

using namespace std;

#define block_circulant
#define CIRCULANT_SIZE 8
template<class Dtype>
class lstm_kernel {
private:

    uint input_size;  //size of the input data
    uint hidden_size; //size of the hidden state
    uint layerIdx;    //layer 0 or layer 1

    /*the pointers to the weights and biases of each gate*/
    Dtype *w_xi = NULL;
    Dtype *w_xf = NULL;
    Dtype *w_xc = NULL;
    Dtype *w_xo = NULL;

    Dtype *w_hi = NULL;
    Dtype *w_hf = NULL;
    Dtype *w_hc = NULL;
    Dtype *w_ho = NULL;

    Dtype *b_i = NULL;
    Dtype *b_f = NULL;
    Dtype *b_c = NULL;
    Dtype *b_o = NULL;

    int w_scale; //scale factor, power of 2
    int b_scale;

public:
    //hidden state and cell state
    //these states will be used and updated at every time step
    Dtype *h_state = NULL;
    Dtype *c_state = NULL;
    bool is_circulant[CIRCULANT_SIZE] = {0}; /*use this mask to indicate what matrices are block circulant* 0-7: w_xi,w_xf,w_xc,w_xo,w_hi,w_hf,w_hc,w_ho*/

    lstm_kernel(const uint input_size, const uint hidden_size, const uint layerIdx) {
        this->input_size = input_size;
        this->hidden_size = hidden_size;
        this->layerIdx = layerIdx;
    }


    Activation<Dtype> * activation = new Activation<Dtype>(std::string("./"));

    ~lstm_kernel() {
        free(w_xi);
        free(w_xf);
        free(w_xc);
        free(w_xo);

        free(w_hi);
        free(w_hf);
        free(w_hc);
        free(w_ho);

        free(b_i);
        free(b_f);
        free(b_c);
        free(b_o);

        free(h_state);
        free(c_state);
    }

    void reset() {

        //clear the hidden state and cell state
        if (!this->h_state) {
            this->h_state = (Dtype *) malloc(sizeof(Dtype) * this->hidden_size);
        }
        if (!this->c_state) {
            this->c_state = (Dtype *) malloc(sizeof(Dtype) * this->hidden_size);
        }
        memset(h_state, 0, sizeof(Dtype) * this->hidden_size);
        memset(c_state, 0, sizeof(Dtype) * this->hidden_size);

    }


    void read_weights_from_file(std::string & weights_file, uint size, Dtype * pweights, bool is_circulant, int scale = 0)
    {
        std::ifstream infile;
        infile.open(weights_file.data());
        assert(infile.is_open());
        std::string s;
        //read shape
        getline(infile, s);

        if(is_circulant)
            size/=CIRCULANT_SIZE;

        for(int i = 0;i<size;i++)
        {
            std::string tmp;
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

    void load_params(std::string &weight_path) {

        cout<<"loading lstm "<<layerIdx<<endl;

        //load scale
        string w_scale_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_w_scale.txt");
        read_scale_from_file(w_scale_file, &this->w_scale);

        string b_scale_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_b_scale.txt");
        read_scale_from_file(b_scale_file, &this->b_scale);

        this->w_xi = (Dtype *) malloc(this->input_size * this->hidden_size * sizeof(Dtype));
        string weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_wxi.txt");
        // IF Using block circulant, we only use 1/CIRCULANT_SIZE of the total allocated memory
        read_weights_from_file(weight_file, this->input_size * this->hidden_size, this->w_xi,this->is_circulant[0], this->w_scale);

        this->w_xf = (Dtype *) malloc(this->input_size * this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_wxf.txt");
        read_weights_from_file(weight_file, this->input_size * this->hidden_size , this->w_xf,this->is_circulant[1],this->w_scale);

        this->w_xc = (Dtype *) malloc(this->input_size * this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_wxc.txt");
        read_weights_from_file(weight_file, this->input_size * this->hidden_size , this->w_xc,this->is_circulant[2],this->w_scale);

        this->w_xo = (Dtype *) malloc(this->input_size * this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_wxo.txt");
        read_weights_from_file(weight_file, this->input_size * this->hidden_size , this->w_xo,this->is_circulant[3],this->w_scale);

        this->w_hi = (Dtype *) malloc(this->hidden_size * this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_whi.txt");
        read_weights_from_file(weight_file, this->hidden_size * this->hidden_size , this->w_hi,this->is_circulant[4],this->w_scale);

        this->w_hf = (Dtype *) malloc(this->hidden_size * this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_whf.txt");
        read_weights_from_file(weight_file, this->hidden_size * this->hidden_size , this->w_hf,this->is_circulant[5],this->w_scale);

        this->w_hc = (Dtype *) malloc(this->hidden_size * this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_whc.txt");
        read_weights_from_file(weight_file, this->hidden_size * this->hidden_size , this->w_hc,this->is_circulant[6],this->w_scale);

        this->w_ho = (Dtype *) malloc(this->hidden_size * this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_who.txt");
        read_weights_from_file(weight_file, this->hidden_size * this->hidden_size , this->w_ho,this->is_circulant[7],this->w_scale);

        this->b_i = (Dtype *) malloc(this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_bi.txt");
        read_weights_from_file(weight_file, this->hidden_size , this->b_i, false,this->b_scale);

        this->b_f = (Dtype *) malloc(this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_bf.txt");
        read_weights_from_file(weight_file, this->hidden_size , this->b_f,false,this->b_scale);

        this->b_c = (Dtype *) malloc(this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_bc.txt");
        read_weights_from_file(weight_file, this->hidden_size , this->b_c,false,this->b_scale);

        this->b_o = (Dtype *) malloc(this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/lstm_")+to_string(layerIdx)+string("_bo.txt");
        read_weights_from_file(weight_file, this->hidden_size , this->b_o,false,this->b_scale);
    }



    void forward(Dtype *x) {

        Dtype *h_t = this->h_state;
        Dtype *c_t = this->c_state;
        /*------------------------Input gate-------------------------------*/
        // w_xi@x
        Dtype *w_xi_mm_x = MatMul(w_xi, x, hidden_size, input_size, input_size, 1, this->is_circulant[0]);

        //w_hi@h_t
        Dtype *w_hi_mm_h_t = MatMul(w_hi, h_t, hidden_size, hidden_size, hidden_size, 1, this->is_circulant[4]);

        // w_xi@x + b_i + w_hi@h_t
        Dtype *i = activation->sigmoid(MatAddTri(w_xi_mm_x, b_i, w_hi_mm_h_t, hidden_size, 1), hidden_size, 1);

        /*-------------------------Forget gate-----------------------------*/

        // w_xf@x
        Dtype *w_xf_mm_x = MatMul(w_xf, x, hidden_size, input_size, input_size, 1,this->is_circulant[1]);


        //w_hf@h_t
        Dtype *w_hf_mm_h_t = MatMul(w_hf, h_t, hidden_size, hidden_size, hidden_size, 1, this->is_circulant[5]);


        // w_xf@x + b_f + w_hf@h_t
        Dtype *f = activation->sigmoid(MatAddTri(w_xf_mm_x, b_f, w_hf_mm_h_t, hidden_size, 1), hidden_size, 1);
        /*-----------------------------Cell gate------------------------------*/

        // w_xc@x
        Dtype *w_xc_mm_x = MatMul(w_xc, x, hidden_size, input_size, input_size, 1,this->is_circulant[2]);

        //w_hc@h_t
        Dtype *w_hc_mm_h_t = MatMul(w_hc, h_t, hidden_size, hidden_size, hidden_size, 1, this->is_circulant[6]);

        // w_xc@x + b_c + w_hc@h_t
        Dtype *g = activation->tanh(MatAddTri(w_xc_mm_x, b_c, w_hc_mm_h_t, hidden_size, 1), hidden_size, 1);

        /*----------------------------Output gate------------------------------*/
        // w_xo@x
        Dtype *w_xo_mm_x = MatMul(w_xo, x, hidden_size, input_size, input_size, 1,this->is_circulant[3]);

        //w_ho@h_t
        Dtype *w_ho_mm_h_t = MatMul(w_ho, h_t, hidden_size, hidden_size, hidden_size, 1, this->is_circulant[7]);

        // w_xo@x + b_o + w_ho@h_t
        Dtype *o = activation->sigmoid(MatAddTri(w_xo_mm_x, b_o, w_ho_mm_h_t, hidden_size, 1), hidden_size, 1);

        /*c_next*/
        //c_next = f * c_t + i * g
        Dtype *f_dot_c_t = MatDot(f, c_t, hidden_size, 1);
        Dtype *i_dot_g = MatDot(i, g, hidden_size, 1);
        Dtype *c_next = MatAddDuo(f_dot_c_t, i_dot_g, hidden_size, 1);
        /*h_next*/
        Dtype *h_next = MatDot(o, activation->tanh(c_next, hidden_size, 1), hidden_size, 1);

        //update the reserved hidden states and  cell states
        memcpy(this->h_state, h_next, sizeof(Dtype) * hidden_size);
        memcpy(this->c_state, c_next, sizeof(Dtype) * hidden_size);

        //free the allocated temporary memory
        free(w_xi_mm_x);
        free(w_hi_mm_h_t);
        free(i);
        free(w_xf_mm_x);
        free(w_hf_mm_h_t);
        free(f);
        free(w_xc_mm_x);
        free(w_hc_mm_h_t);
        free(g);
        free(w_xo_mm_x);
        free(w_ho_mm_h_t);
        free(o);
        free(f_dot_c_t);
        free(i_dot_g);
        free(h_next);
        free(c_next);
    }
};


#endif //LSTM_LSTM_H

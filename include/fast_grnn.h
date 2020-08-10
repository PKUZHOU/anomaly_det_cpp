//
// Created by zhezhou on 2020/8/10.
//

#ifndef LSTM_FAST_GRNN_H
#define LSTM_FAST_GRNN_H
//
// Created by zhou on 2019/9/17.
//

#include "operator.h"
#include <string>
#include <fstream>

using namespace std;

#define block_circulant
#define CIRCULANT_SIZE 8
template<class Dtype, class T_temp>
class fast_grnn_kernel {
private:

    uint input_size;  //size of the input data
    uint hidden_size; //size of the hidden state
    uint layerIdx;    //layer 0 or layer 1

    /*the pointers to the weights and biases of each gate*/
    Dtype *W = NULL;

    Dtype *U = NULL;

    Dtype *b_z = NULL;
    Dtype *b_h = NULL;

    Dtype * zeta = NULL;
    Dtype * nu = NULL;

    // S/2^n
    T_temp S; int n;

public:
    //hidden state and cell state
    //these states will be used and updated at every time step
    Dtype *h_state = NULL;

    bool is_circulant[CIRCULANT_SIZE] = {0}; /*use this mask to indicate what matrices are block circulant*/

    fast_grnn_kernel(const uint input_size, const uint hidden_size, const uint layerIdx) {
        this->input_size = input_size;
        this->hidden_size = hidden_size;
        this->layerIdx = layerIdx;
    }


    Activation<Dtype> * activation = new Activation<Dtype>(std::string("./"));

    ~fast_grnn_kernel() {
        free(W);
        free(U);
        free(b_z);
        free(b_h);
        free(zeta);
        free(nu);

        free(h_state);
    }

    void reset() {

        //clear the hidden state and cell state
        if (!this->h_state) {
            this->h_state = (Dtype *) malloc(sizeof(Dtype) * this->hidden_size);
        }
        memset(h_state, 0, sizeof(Dtype) * this->hidden_size);
    }


    void read_weights_from_file(std::string & weights_file, uint size, Dtype * pweights, bool is_circulant)
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
        }
    }

    void read_scale_from_file(std::string & scale_file)
    {
        std::ifstream infile;
        infile.open(scale_file.data());
        assert(infile.is_open());
        std::string s;
        getline(infile, s);
    }

    void load_params(std::string &weight_path) {

        cout<<"loading fast grnn "<<layerIdx<<endl;

        this->W = (Dtype *) malloc(this->input_size * this->hidden_size * sizeof(Dtype));
        string weight_file = weight_path+string("/fast_grnn_")+to_string(layerIdx)+string("_W.txt");
        // IF Using block circulant, we only use 1/CIRCULANT_SIZE of the total allocated memory
        read_weights_from_file(weight_file, this->input_size * this->hidden_size, this->W);

        weight_file = weight_path+string("/fast_grnn_")+to_string(layerIdx)+string("_U.txt");
        read_weights_from_file(weight_file, this->hidden_size * this->hidden_size , this->U);

        this->b_z = (Dtype *) malloc(this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/fast_grnn_")+to_string(layerIdx)+string("_b_z.txt");
        read_weights_from_file(weight_file, this->hidden_size , this->b_z, false);

        this->b_h = (Dtype *) malloc(this->hidden_size * sizeof(Dtype));
        weight_file = weight_path+string("/fast_grnn_")+to_string(layerIdx)+string("_b_h.txt");
        read_weights_from_file(weight_file, this->hidden_size , this->b_h, false);
    }


    void forward(Dtype *x_t) {

        Dtype *h_t = this->h_state;

        // W@x
        T_temp *W_mm_x = MatMul(W, x_t, hidden_size, input_size, input_size, 1, this->is_circulant[0]);

        // U@h
        T_temp *U_mm_h_t = MatMul(U, h_t, hidden_size, hidden_size, hidden_size, 1, this->is_circulant[1]);

        // sigma(W@x+U@h+b_z)
        Dtype *z_t = activation->sigmoid(MulShift<Dtype,T_temp>(MatAddTri(W_mm_x, b_z, U_mm_h_t, hidden_size, 1),S,n), hidden_size, 1);

        // tanh(W@x+U@h+b_h)
        Dtype *h_hat = activation->tanh(MulShift<Dtype,T_temp>(MatAddTri(W_mm_x, b_h, U_mm_h_t, hidden_size, 1),S,n), hidden_size, 1);

        // 1-z_t
        Dtype * one_sub_z_t = ScalaSub(Dtype(1.0), z_t, hidden_size, 1);

        //zeta(1-z_t)
        T_temp * zeta_mul = ScalaMul(zeta, one_sub_z_t, hidden_size, 1);

        //zeta(1-z_t)+nu
        T_temp * c_b = ScalaAdd(nu, zeta_mul, hidden_size, 1);

        //(zeta(1-z_t)+nu)dot(h_hat)
        T_temp * c_b_dot_h_hat = MatDot(c_b, h_hat, hidden_size, 1);

        //z_t dot h_t
        T_temp * z_t_dot_h_t = MatDot(z_t, h_t, hidden_size, 1);


        Dtype * h_next = MulShift<Dtype, T_temp>(MatAddDuo(c_b_dot_h_hat, z_t_dot_h_t, hidden_size, 1),S,n);

        //update the reserved hidden states and  cell states
        memcpy(this->h_state, h_next, sizeof(Dtype) * hidden_size);

        //free the allocated temporary memory
        free(h_hat);
        free(c_b_dot_h_hat);
        free(c_b);
        free(z_t_dot_h_t);
        free(z_t);
        free(W_mm_x);
        free(U_mm_h_t);
        free(h_next);
        free(h_t);
    }
};


#endif //LSTM_FAST_GRNN_H

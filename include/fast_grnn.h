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
#include "data_type.h"

using namespace std;

#define  BLOCK_SIZE 8

template <class Dtype>
class fast_grnn_cell {
//private:
public:
    uint input_size;  //size of the input data
    uint hidden_size; //size of the hidden state
    uint layerIdx;    //layer 0 or layer 1

    /*the pointers to the weights and biases of each gate*/
    Mat<Dtype>* W = NULL;
    Mat<Dtype>* U = NULL;
    Mat<Dtype>* bias_gate = NULL;
    Mat<Dtype>* bias_update = NULL;
    Mat<Dtype>* zeta = NULL;
    Mat<Dtype>* nu = NULL;

    /*integer bits*/
    uint S_in = 0;
    uint S_wComp = 0;
    uint S_uComp = 0;

    uint S_z_state = 0;
    uint S_c_b = 0;
    uint S_c_af = 0;

    uint S_W = 0;
    uint S_U = 0;
    uint S_zeta = 0;
    uint S_nu = 0;
    uint S_bias_gate = 0;
    uint S_bias_update = 0;

    fast_grnn_cell(const uint input_size, const uint hidden_size, const uint layerIdx) {
        this->input_size = input_size;
        this->hidden_size = hidden_size;
        this->layerIdx = layerIdx;
    }

    // TODO : new activation table
    Activation<Dtype> * activation = new Activation<Dtype>(std::string("./"));

    ~fast_grnn_cell() {
        delete (W);
        delete (U);
        delete (bias_gate);
        delete (bias_update);
        delete (zeta);
        delete (nu);
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

    void read_scale_from_file(std::string & scale_file, uint & p_scale){
        std::ifstream infile;
        infile.open(scale_file.data());
        assert(infile.is_open());
        infile >> p_scale;
    }

    void load_scale(std::string &scale_path){
        cout <<"loading scale"<<layerIdx<<endl;

        #define LOAD(file_name, data)  \
                scale_file = scale_path+string("/")+string(file_name)+to_string(layerIdx)+string(".txt"); \
                read_scale_from_file(scale_file,data);

        string scale_file;
        LOAD("in",          this->S_in)
        LOAD("wComp",       this->S_wComp)
        LOAD("uComp",       this->S_uComp)
        LOAD("z_state",     this->S_z_state)
        LOAD("c_b",         this->S_c_b)
        LOAD("c_af",        this->S_c_af)
        LOAD("W",           this->S_W)
        LOAD("U",           this->S_U)
        LOAD("zeta",        this->S_zeta)
        LOAD("nu",          this->S_nu)
        LOAD("bias_gate",   this->S_bias_gate)
        LOAD("bias_update", this->S_bias_update)
    }

    void load_weight(std::string &weight_path){

        #define LOAD(data, row, col, scale, file_name) \
                data = new Mat<Dtype>(row,  col, scale); \
                weight_file = weight_path+ string("/") + string(file_name) +to_string(layerIdx)+string(".txt"); \
                read_weights_from_file(weight_file, (row) * col, data);

        string weight_file;
        if(layerIdx == 0){
            LOAD(this->W, this->hidden_size, this->input_size, S_W, "W")
        }
        else{
            LOAD(this->W, this->hidden_size/BLOCK_SIZE, this->input_size,  S_W, "W")
        }

        LOAD(this->U, this->hidden_size/BLOCK_SIZE, this->hidden_size, S_U, "U")

        LOAD(this->bias_update, this->hidden_size, 1 , S_bias_update,"bias_update")
        LOAD(this->bias_gate, this->hidden_size, 1, S_bias_gate,"bias_gate")
        LOAD(this->nu,1,1,S_nu ,"nu")
        LOAD(this->zeta, 1, 1, S_zeta, "zeta")
    }

    void load_params(std::string &param_path) {
        cout<<"loading fast grnn "<<layerIdx<<endl;
        string weight_path = param_path + string("/weight");
        string scale_path = param_path + string("/scale");
        load_scale(scale_path);
        load_weight(weight_path);
    }

    void forward(Mat<Dtype> *x_t, Mat<Dtype> * h_state) {
        Mat<Dtype> *h_t = h_state;

        Mat<Dtype> *wComp;

        if(layerIdx == 0){
            wComp = W->matmul(x_t,S_wComp);
        }
        else{
            wComp = W->circ_mul(x_t,S_wComp);
        }


        Mat<Dtype> *uComp = U->circ_mul(h_t,S_uComp);

        Mat<Dtype> *preComp = wComp->matadd(uComp);


        Mat<Dtype> *z_s_before = preComp->matadd(bias_gate);
        Mat<Dtype> *z = activation->sigmoid(z_s_before);
        Mat<Dtype> *c_t_before = preComp->matadd(bias_update);
        Mat<Dtype> *c = activation->tanh(c_t_before);
        Mat<Dtype> *one_z = z->one_sub();
        Mat<Dtype> *z_state = z->mul(h_t, S_z_state);
        Mat<Dtype> *pre_c_b = one_z->matmul(zeta,S_c_b);
        Mat<Dtype> *c_b = pre_c_b->add_scalar(nu);
        Mat<Dtype> *c_af = c_b->mul(c, S_c_af);
        Mat<Dtype> *new_h = z_state->matadd(c_af);
        *(h_state) = *new_h;

//        cout<<layerIdx<<endl;
//        cout<<"-----------------------"<<endl;
//        for(int i =0;i<64;i++){
//            cout<<(*h_state)[i]<<" ";
//        }
//        cout<<endl;

    }
};


#endif //LSTM_FAST_GRNN_H

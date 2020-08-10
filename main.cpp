#include <iostream>
#include "model.h"
#include "data_type.h"
#include "operator.h"
using namespace std;

int test_fast_grnn(){
    cout << "Initialization" << std::endl;
    string weights_path = "./fast_grnn_weights_int8";

    const uint input_size = 1;
    const uint hidden_size =64;
    const uint output_size = 1;
    const uint time_steps = 20;

    auto model = new FAST_GRNN<INT<8>, INT<20>>(input_size, hidden_size, output_size, weights_path);
    INT<8> input[time_steps] = {
        #include "input_data_1.h"
    };

    cout << "Forwarding" << std::endl;
    model->forward(input,time_steps);

    cout<<"output is :";
    for(int i = 0;i<output_size;i++)
    cout<<model->out[i]<<" " <<model->out[i];
    cout<<endl;

    return 0;
}

int main() {
    test_fast_grnn();
}
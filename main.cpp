#include <iostream>
#include "model.h"
#include "data_type.h"
#include "operator.h"
using namespace std;
//

int test_int16(){
    cout << "Initialization" << std::endl;
    string weights_path = "./block_circulant/fix_16_weights";

    const uint input_size = 1;
    const uint hidden_size =64;
    const uint output_size = 1;
    const uint time_steps = 20;

    auto model = new Model<Fixed_point<16,3,13>>(input_size, hidden_size, output_size, weights_path);
    Fixed_point<16,3,13> input[time_steps] = {
    #include "input_data_1.h"
    };

    cout << "Forwarding" << std::endl;
    model->forward(input,time_steps);

    cout<<"output is :";
    for(int i = 0;i<output_size;i++)
    cout<<model->out[i]<<" " <<float(model->out[i]);
    cout<<endl;

    return 0;
}

int main() {
    test_int16();
}

//int main() {
//    Activation <Fixed_point<9,3,6>> act;
//    act.generate_tanh_table();
//}
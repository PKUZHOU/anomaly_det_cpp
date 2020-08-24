#include <iostream>
#include "model.h"
#include "data_type.h"
#include "operator.h"
using namespace std;

typedef INT<8> INT8;

int test_fast_grnn(){
    cout << "Initialization" << std::endl;
    string param_path = "./txtfile";

    const uint input_size = 1;
    const uint hidden_size =64;
    const uint output_size = 5;
    const uint time_steps = 20;

    auto model = new FAST_GRNN<INT8>(input_size, hidden_size, output_size, param_path);

    INT8 input_data[time_steps] = {
            "18","4D","03","C3","AF","22","43","3A","C3","AD","3E","53","2B","BC","C7","41","4F","23","AB","D0",
    };

    cout << "Forwarding" << std::endl;
    model->forward(input_data,time_steps);

    cout<<"output is :";
    for(int i = 0;i<output_size;i++)
    cout<<(*model->out)[i]<<" ";
    cout<<endl;
    return 0;
}

int main() {
    test_fast_grnn();
}
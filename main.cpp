#include <iostream>
#include "model.h"
#include "data_type.h"
#include "operator.h"
using namespace std;

typedef INT<16> INT16;

int test_fast_grnn(){
    cout << "Initialization" << std::endl;
    string param_path = "./txtfile_new";

    const uint input_size = 1;
    const uint hidden_size =128;
    const uint output_size = 5;
    const uint time_steps = 256;

    auto model = new FAST_GRNN<INT16>(input_size, hidden_size, output_size, param_path);

    // INT16 input_data[time_steps] = {
    //         "18","4D","03","C3","AF","22","43","3A","C3","AD","3E","53","2B","BC","C7","41","4F","23","AB","D0",
    // };
/*
    INT16 input_data[time_steps] = {
            "0C2D","2699","0193","E125","D70D","1101","21F5","1D1F","E111","D667","1F19","29E1","15C1","DD9F","E301","20A3","27DF","11BF","D52D","E793",
    };
*/
    INT16 input_data[time_steps] = {
        #include "txtfile_new/inputs.h"
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

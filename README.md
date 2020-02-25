#LSTM Model For Anomaly Detection

This is an anomaly detection model which is composed of a two-layer LSTM and a    Fully connected layer

You can run this project by running the following commands:
`cmake .`
`make -j`
`./lstm`

#### 2019-12-30 Update

In this version, I replace the 64x64 matrices with so-called `block circulant matrix`, thus saving 7/8 of memory usage. 
I use this bool array `bool is_circulant[CIRCULANT_SIZE]` in each lstm layer to identify the block circulant matrices, and you can find how it computes in the `MatMul()` function in `operator.h`.
I generate new weight and spread sheet, you can find them in the `block_circulant` folder. When you run this c++ code, you will get:

    output is :DF99 -1.01257


(In fact my C++ version of block circulant matrices computing looks a little ugly now, you may find a better way to do it when designing the hardware logic).



#### 2019-12-9 Update
In this version, I implement the 16bit fixed-point sample using C++, and have verified the consistency with our FPGA implementation.
If you run this code, you will get the output:
    
    BF50 -2.02148

The BF50 is the the 16bit fixed-point number in hexadecimal, and -2.02148 is its float value.

#### Note that:

In this folder, I also provide some useful data including folder `fix_16_weights`, folder `spread_sheet` , `tanh_table.txt` and `sigmoid_table.txt`. 
* In the folder `fix_16_weights`  you will see the weights stored in txt files. eg. `lstm_0_bc.txt`,`lstm_1_wxi.txt`, the prefix `lstm` or `fc` indicates its type, the number `0` or `1` indicates its layer index, and the `bc` shows it is the `bias` of gate `c` and `wxi` indicates it is the weight of matrix `xi` 

* In `spread_sheet` folder you will see the dumped data during running the sample. All the data is stored in hex format so you can easily varify your hardware design use these data.  The file names start with its step, eg. `step_12_lstm_0_w_xi@x` indicates that it is dumped during the 12th step (20 steps in total). `lstm_0_w_xi@x` indicates that it is the output of `w_xi` mutiplies `x` in the first lstm layer.

* `tanh_table.txt` and `sigmoid_table.txt` store the look-up table of tanh and sigmoid function, each has 512 entries.
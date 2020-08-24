#LSTM Model For Anomaly Detection

This is an anomaly detection model which is composed of a two-layer LSTM and a    Fully connected layer

You can run this project by running the following commands:
`cmake .`
`make -j`
`./lstm`

#### 2020-8-10 Update

Add fastGRNN (int8)


#### 2020-3-3 Update

Now we have regenerated the weight and spreadsheet, which you can see in the block_circulant folder 
The outputs are verified, and the average error rate compared with the original method is just 0.1%. 

The output should be

    `output is :D6E6 -1.28442`

#### 2020-2-27 Update

I modify the generation logic of look-up table, then remove the multiplication in look-up operations.
Now you can directly get the corresponding value using the input without multiplication.

The output should be

   `output is :DC0A -1.12378`

Slightly different from the former value due to the modification of computational logic, I am going to check if this difference has any affects, and generate the new spreadsheet

#### 2020-2-25 Update

I have rewritten the acivation function.
The output is slightly different from the original version, now you will get

   `output is :DF38 -1.01526`
The difference is caused by the modification of computational logic, and should have little effect.

If necessary, I will re-generate the spread_sheet later.

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

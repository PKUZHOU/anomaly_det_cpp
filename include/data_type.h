//
// Created by zhou on 2019/12/6.
//

#ifndef LSTM_DATA_TYPE_H
#define LSTM_DATA_TYPE_H
#include <iomanip>
#include <iostream>
#include <algorithm>
typedef unsigned short u16;

template <int tot_bits = 8>
class INT{
public:
    int data = 0; // use int32 to hold the fixed_point data
    int TOTBits = 8;
    int saturate(int b)  // truncate the fixed_point data to the valid range.
    {
        if(b> ((1<<(TOTBits-1))-1))
            return ((1<<(TOTBits-1))-1);
        if(b<-(1<<(TOTBits-1)))
            return -(1<<(TOTBits-1));
        return b;
    }
    INT() {}
    INT(const char raw[2]) // init from hex data , eg. "1F3E"
    {
        for(int i = 0;i<2;i++)
        {
            if('0'<=raw[i] and raw[i]<='9')
            {
                data+=((raw[i]-'0')<<(2*(1-i)));
            }
            else
            {
                data+=((raw[i]-'A'+10)<<(2*(1-i)));
            }
        }
        data = data<<(32-TOTBits)>>(32-TOTBits); // the shift operation guarantees the two's complement format for negative values
    }

    INT(const INT& b) //init from other fixed_point data
    {
        data = b.data;
        TOTBits = b.TOTBits;
    }

    //TODO add, mul, sub

    // print the fixed point data in hex format
    friend std::ostream& operator <<(std::ostream& os, const INT& b) {
        int b_data = b.data;

        if(b.TOTBits == 16){
            b_data = b_data&0xFFFF;
            for(int i = 0;i<4;i++) {
                int tmp = ((b_data >> (4 * (3 - i))) & 0xF);
                if (tmp <= 9) {
                    os << char(tmp + '0');
                } else {
                    os << char(tmp - 10 + 'A');
                }
            }
        }
        else if(b.TOTBits == 8){
            b_data = b_data&0xFF;
            for(int i = 0;i<2;i++) {
                int tmp = ((b_data >> (2 * (1 - i))) & 0xF);
                if (tmp <= 9) {
                    os << char(tmp + '0');
                } else {
                    os << char(tmp - 10 + 'A');
                }
            }
        }

        return os;
    }

    // fill it using std::in
    // eg: cin>>b
    friend std::istream& operator >>(std::istream& is,  INT& b) {

        std::string tmp;
        is>>tmp;
        transform(tmp.begin(),tmp.end(),tmp.begin(),::toupper);
        b = INT(tmp.c_str());
        b.SCALE = 13;
        return is;
    }

};


#endif //LSTM_DATA_TYPE_H

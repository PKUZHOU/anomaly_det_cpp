//
// Created by zhou on 2019/12/6.
//

#ifndef LSTM_DATA_TYPE_H
#define LSTM_DATA_TYPE_H
#include <iomanip>
#include <iostream>
#include <algorithm>
typedef unsigned short u16;

template <int totBits=16, int intBits = 3, int decBits=13>//The default config is Fix_16<3,13>
class Fixed_point{
public:
    int data = 0; // use int32 to hold the fixed_point data
    int TOTBits = totBits;
    int INTBits = intBits;
    int DECBits = decBits;

    int saturate(int b)  // truncate the fixed_point data to the valid range.
    {
        if(b> ((1<<(TOTBits-1))-1))
            return ((1<<(TOTBits-1))-1);
        if(b<-(1<<(TOTBits-1)))
            return -(1<<(TOTBits-1));
        return b;
    }

    Fixed_point() {}
    Fixed_point(const char raw[4]) // init from hex data , eg. "1F3E"
    {
        for(int i = 0;i<4;i++)
        {
            if('0'<=raw[i] and raw[i]<='9')
            {
                data+=((raw[i]-'0')<<(4*(3-i)));
            }
            else
            {
                data+=((raw[i]-'A'+10)<<(4*(3-i)));
            }
        }
        data = data<<(32-TOTBits)>>(32-TOTBits); // the shift operation guarantees the two's complement format for negative values
    }

    Fixed_point(const Fixed_point& b) //init from other fixed_point data
    {
        int b_DECBits = b.DECBits;
        int b_data = b.data;

        // align the decimal
        if(b_DECBits >= DECBits)
        {
            data = saturate(b_data >>(b_DECBits-DECBits));
        }
        else
        {
            data = saturate(b_data <<(DECBits-b_DECBits));
        }
        data = b.data;
    }

    //init from float number
    Fixed_point(const float &b)
    {
        data = saturate(int(b*(1<<DECBits)));
    }

    //init from int number
    Fixed_point(const int &b)
    {
        data = saturate(int(b*(1<<DECBits)));
    }

    //c = a + b
    Fixed_point operator + (const Fixed_point &b)
    {
        Fixed_point c;

        int a_data = data;
        int b_data = b.data;
        int c_data = saturate(a_data+b_data);

        c.data = c_data;
        return c;
    }

    // eg. c =  a + 1.0
    Fixed_point operator + (const float &b)
    {
        Fixed_point c;
        int a_data = data;

        Fixed_point fix_b(b);

        int b_data = fix_b.data;
        int c_data = saturate(a_data+b_data);

        c.data = c_data;
        return c;
    }

    //eg. c = 1.0 + b
    friend Fixed_point operator + (const float &a, const Fixed_point b)
    {
        Fixed_point c;
        Fixed_point fix_a(a);
        int a_data = fix_a.data;
        int b_data = b.data;
        int c_data = fix_a.saturate(a_data+b_data);

        c.data = c_data;
        return c;
    }

    //eg. a+=b
    Fixed_point operator += (const Fixed_point &b)
    {

        int b_data = b.data;
        int c_data = saturate(b_data+ data);
        data = c_data;
        return *this;
    }

    //eg. c =  a-b
    Fixed_point operator - (const Fixed_point &b)
    {
        Fixed_point c;
        int a_data = data;
        int b_data = b.data;
        int c_data = saturate(a_data-b_data);
        c.data = c_data;
        return c;
    }

    //eg. c = -a
    Fixed_point operator - ()
    {
        Fixed_point c;
        int a_data = 0;
        int b_data = data;
        int c_data = saturate(a_data-b_data);
        c.data = c_data;
        return c;
    }
    //eg. c = a*b
    Fixed_point operator * (const Fixed_point &b)
    {
        Fixed_point c;

        int a_data = data;
        int b_data = b.data;
        int c_data = saturate(a_data*b_data>>DECBits);

        c.data = c_data;
        return c;
    }
    //eg. a*=b
    Fixed_point operator *= (const Fixed_point &b)
    {

        int b_data = b.data;
        int c_data = saturate(b_data*data>>DECBits);
        data = c_data;
        return *this;
    }


    // print the fixed point data in hex format
    friend std::ostream& operator <<(std::ostream& os, const Fixed_point& b) {
        int b_data = b.data;
        b_data = b_data&0xFFFF;
        for(int i = 0;i<4;i++)
        {
            int tmp = ((b_data>>(4*(3-i)))&0xF);
            if(tmp<=9)
            {
                os<<char(tmp+'0');
            }
            else
            {
                os<<char(tmp-10+'A');
            }
        }
        return os;
    }

    // fill it using std::in
    // eg: cin>>b
    friend std::istream& operator >>(std::istream& is,  Fixed_point& b) {

        std::string tmp;
        is>>tmp;
        transform(tmp.begin(),tmp.end(),tmp.begin(),::toupper);
        b = Fixed_point(tmp.c_str());
        return is;
    }

    // convert to float data
    explicit  operator float()
    {
        int tmp = data;
        tmp = tmp << (32-TOTBits);
        tmp = tmp >> (32-TOTBits);
        return float(tmp)/(1<<DECBits);
    }

};


#endif //LSTM_DATA_TYPE_H

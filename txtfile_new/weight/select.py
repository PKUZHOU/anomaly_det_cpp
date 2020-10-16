import os
out = open("fc_w_2.txt",'w')
with open("fc_w.txt",'r') as f:
    data = f.readlines()[1:]
    for i in range(128):
        for j in range(2):
            index = i * 5 + j
            out.write(data[index])



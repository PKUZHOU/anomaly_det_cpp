import os 
#files = os.listdir("inputs")
#files = sorted(files)

inputs_h = open("inputs.h",'w')

for i in range(256):
    with open("inputs/step_{}_input_0.txt".format(i),'r') as f:
        data = f.readlines()
        x = data[1].strip()
        inputs_h.write("\"{}\",\n".format(x))

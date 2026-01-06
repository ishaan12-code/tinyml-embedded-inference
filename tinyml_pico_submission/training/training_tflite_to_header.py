import sys

data = open(sys.argv[1],"rb").read()
name = sys.argv[3]

with open(sys.argv[2],"w") as f:
    f.write("#pragma once\n#include <cstdint>\n")
    f.write(f"alignas(16) const unsigned char {name}[]={{\n")
    for i,b in enumerate(data):
        if i%12==0: f.write(" ")
        f.write(f"0x{b:02x},")
        if i%12==11: f.write("\n")
    f.write("\n};\n")
    f.write(f"const unsigned int {name}_len={len(data)};\n")

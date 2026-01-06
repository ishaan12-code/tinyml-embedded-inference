import numpy as np

m = np.load("artifacts/mean.npy").reshape(-1)
s = np.load("artifacts/std.npy").reshape(-1)

F = len(m)

def fmt(arr):
    return ", ".join([f"{x:.8f}f" for x in arr])

text = []
text.append("#pragma once\n")
text.append(f"constexpr int kNumFeatures = {F};\n\n")
text.append(f"static const float g_mean[kNumFeatures] = {{ {fmt(m)} }};\n\n")
text.append(f"static const float g_std[kNumFeatures]  = {{ {fmt(s)} }};\n")

with open(r"..\pico_firmware\mean_std.h", "w", encoding="utf-8") as f:
    f.writelines(text)

print("WROTE ..\\pico_firmware\\mean_std.h  F =", F)

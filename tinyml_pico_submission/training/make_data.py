import numpy as np

np.random.seed(42)

N = 1000   # number of samples
F = 20     # number of features

X0 = np.random.normal(0.0, 1.0, (N//2, F))
y0 = np.zeros(N//2, dtype=np.int32)

X1 = np.random.normal(1.0, 1.0, (N//2, F))
y1 = np.ones(N//2, dtype=np.int32)

X = np.vstack([X0, X1]).astype(np.float32)
y = np.concatenate([y0, y1])

idx = np.random.permutation(N)
X = X[idx]
y = y[idx]

np.save("X.npy", X)
np.save("y.npy", y)

print("DONE")
print("X shape:", X.shape)
print("y shape:", y.shape)

import os, time
import numpy as np
import keras

# Optional: silence oneDNN warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

mean = np.load("artifacts/mean.npy").reshape(-1)
std  = np.load("artifacts/std.npy").reshape(-1)

# Keras 3: load SavedModel for inference
layer = keras.layers.TFSMLayer("saved_model", call_endpoint="serve")

MCU_LATENCY_US = 450
MCU_RAM_LIMIT_KB = 64

print("=== Embedded ML Demo (Simulated MCU) ===")
print("RAM budget:", MCU_RAM_LIMIT_KB, "KB")


xw = np.zeros((1, len(mean)), dtype=np.float32)
_ = layer(xw).numpy()

# --- demo runs with non-trivial inputs ---
rng = np.random.default_rng(42)

NUM_RUNS = 20000

for i in range(NUM_RUNS):

    # create "sensor-like" features: mean + noise
    x = (mean + rng.normal(0, 1.0, size=mean.shape).astype(np.float32) * std).reshape(1, -1)

    # normalize like training
    xn = (x - mean) / (std + 1e-8)

    t0 = time.perf_counter_ns()
    y = float(layer(xn).numpy()[0, 0])
    time.sleep(MCU_LATENCY_US / 1e6)   # simulate MCU latency
    t1 = time.perf_counter_ns()

    latency_us = (t1 - t0) // 1000
    pred = 1 if y >= 0.5 else 0

    print(f"run={i} score={y:.4f} pred={pred} latency_us≈{int(latency_us)}")


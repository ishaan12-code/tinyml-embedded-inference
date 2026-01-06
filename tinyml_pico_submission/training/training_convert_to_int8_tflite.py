import numpy as np, tensorflow as tf, os

X = np.load("X.npy").astype(np.float32)
mean = np.load("artifacts/mean.npy")
std = np.load("artifacts/std.npy")
X = (X - mean) / std

def rep():
    for i in range(min(200, len(X))):
        yield [X[i:i+1]]

conv = tf.lite.TFLiteConverter.from_saved_model("saved_model")
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8

tfl = conv.convert()
open("artifacts/model_int8.tflite","wb").write(tfl)

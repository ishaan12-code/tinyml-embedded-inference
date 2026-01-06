import os, json, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

X = np.load("X.npy").astype(np.float32)
y = np.load("y.npy").astype(np.int32)

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

mean = Xtr.mean(axis=0, keepdims=True)
std = Xtr.std(axis=0, keepdims=True) + 1e-8
Xtr = (Xtr - mean) / std
Xte = (Xte - mean) / std

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

model.fit(Xtr, ytr, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

probs = model.predict(Xte).ravel()
preds = (probs >= 0.5).astype(int)

acc = accuracy_score(yte, preds)
auc = roc_auc_score(yte, probs)

os.makedirs("artifacts", exist_ok=True)
model.export("saved_model")
np.save("artifacts/mean.npy", mean.astype(np.float32))
np.save("artifacts/std.npy", std.astype(np.float32))
json.dump({"accuracy": float(acc), "auc": float(auc)},
          open("artifacts/test_metrics.json","w"), indent=2)

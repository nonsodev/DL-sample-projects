import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

df = pd.read_csv("Titanic-Dataset.csv")

df = df.drop(["PassengerId", "Name", "Ticket", "Embarked", "Cabin"], axis=1)

df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "male" else 0) # Convert sex to numerical

df = df.dropna() # Drop missing age...

X, y = df.drop("Survived", axis=1), df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights)) # Weights for imbalanced classes


model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(6,), activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(50, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(1, activation="sigmoid")
])

tb_callbacks = tf.keras.callbacks.TensorBoard(log_dir="logs/", histogram_freq=1)


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[tb_callbacks, tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)] # Early stopping
)


evaluation_result = model.evaluate(X_test_scaled, y_test)

print("Evaluation Result:", evaluation_result)
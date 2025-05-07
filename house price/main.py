import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd


df = pd.read_csv("house_price_regression_dataset.csv")
df.info()

df.head()

df.describe()

X, y = df.drop("House_Price", axis=1), df["House_Price"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape(-1,1))
y_test_scaled = scaler_y.transform(y_test.to_numpy().reshape(-1,1)) # Scale the target


model = keras.Sequential([

    keras.layers.Dense(64, input_shape=(X_train_scaled.shape[1],), activation='relu'),

    keras.layers.Dense(32, activation='relu'),

    keras.layers.Dense(16, activation='relu'),
    

    keras.layers.Dense(1, activation=None)  
])

model.compile(optimizer="adam",loss="mean_squared_error", metrics=[
        "mae", 
        tf.keras.metrics.RootMeanSquaredError(),
        tf.keras.metrics.MeanAbsolutePercentageError()
    ])

tb_callbacks = tf.keras.callbacks.TensorBoard(log_dir="logs/")
early_stop = tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
model.fit(X_train_scaled, y_train_scaled, batch_size=32,validation_split=0.2,callbacks=[tb_callbacks,early_stop],epochs=500) # Train the model

model.evaluate(X_test_scaled, y_test_scaled)

y_pred_scaled = model.predict(X_test_scaled)
y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test_scaled)

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

mae_real = mean_absolute_error(y_test_actual, y_pred_actual)
mape_real = mean_absolute_percentage_error(y_test_actual, y_pred_actual)
mse_real = mean_squared_error(y_test_actual,y_pred_actual)

print("Real MAE:", mae_real)
print("Real MAPE:", mape_real)
print("Real MSE", mse_real)
print("Real RSME", np.sqrt(mse_real))
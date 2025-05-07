# House Price Prediction Model

This project aims to predict house prices using a neural network built with TensorFlow and Keras. The model is trained on a dataset containing various features of houses.

## Overview

The goal is to develop a regression model that can accurately predict house prices based on features such as square footage, number of bedrooms and bathrooms, year built, lot size, garage size, and neighborhood quality.

## Data

The dataset used is `house_price_regression_dataset.csv`, which includes features related to houses and their corresponding prices.

## Libraries Used

* `tensorflow`: For building and training the neural network.
* `keras`: The high-level API of TensorFlow for building neural networks.
* `numpy`: For numerical operations.
* `pandas`: For data manipulation and reading the CSV file.
* `sklearn`: For data preprocessing (train-test split, scaling) and evaluation metrics.

## Data Preprocessing

The following steps were performed to prepare the data for training:

1.  **Loading Data:** The dataset was loaded using pandas.
2.  **Feature Separation:** The features (X) and the target variable (y, house prices) were separated.
3.  **Data Splitting:** The data was split into training (80%) and testing (20%) sets using `train_test_split`.
4.  **Feature Scaling:** `MinMaxScaler` was used to scale both the features (X) and the target variable (y) to a range between 0 and 1. Scaling is crucial for neural networks to ensure stable and efficient training.  Note that `y` was reshaped to be a 2D array before scaling.

## Model Architecture

The neural network architecture consists of the following layers:

1.  A dense layer with 64 neurons and ReLU activation. This is the first layer and its `input_shape` is defined based on the number of features in the training data.
2.  A dense layer with 32 neurons and ReLU activation.
3.  A dense layer with 16 neurons and ReLU activation.
4.  A final dense layer with 1 neuron and no activation function.  Since this is a regression task, no activation is used in the output layer.

## Training

The model was trained using the following configurations:

* **Optimizer:** Adam
* **Loss Function:** Mean Squared Error (MSE), a standard loss function for regression problems.
* **Metrics:**
    * Mean Absolute Error (MAE)
    * Root Mean Squared Error (RMSE)
    * Mean Absolute Percentage Error (MAPE)
* **Validation Split:** 20% of the training data was used for validation during training.
* **Epochs:** 500
* **Batch Size:** 32
* **Callbacks:**
    * `TensorBoard`: For logging and visualization of training progress. Logs are saved in the `logs/` directory.
    * `EarlyStopping`: To prevent overfitting and save training time. Training stops if the validation loss doesn't improve for 5 epochs, and the best weights are restored.

## Evaluation

The trained model was evaluated on the test set. The evaluation metrics (loss, MAE, RMSE, MAPE) are calculated on the scaled data.  The model's predictions are then inverse-transformed to the original scale, and MAE, MAPE, MSE, and RMSE are calculated on the actual house price values.

## How to Run

1.  Ensure you have the necessary libraries installed:

    ```bash
    pip install tensorflow numpy pandas scikit-learn
    ```

2.  Place the `house_price_regression_dataset.csv` file in the same directory as the Python script.

3.  Run the script:

    ```bash
    python your_script_name.py
    ```

4.  To view TensorBoard logs:

    ```bash
    tensorboard --logdir logs
    ```

## Results

The script outputs the evaluation metrics on both the scaled and original data, providing a comprehensive view of the model's performance.  Key metrics include MAE (average absolute difference between predicted and actual prices), MAPE (average percentage error), and RMSE (square root of the average squared error).
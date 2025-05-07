# Titanic Survival Prediction

This project utilizes a neural network built with TensorFlow and Keras to predict the survival of passengers on the Titanic. The model is trained on the well-known Titanic dataset.

## Overview

The goal is to build a binary classification model that can predict whether a passenger survived the Titanic disaster based on various features like age, sex, passenger class, etc.

## Data

The dataset used is `Titanic-Dataset.csv`, which contains information about passengers on the Titanic, including whether they survived or not.

## Libraries Used

* `tensorflow`: For building and training the neural network.
* `keras`: The high-level API of TensorFlow for building neural networks.
* `numpy`: For numerical operations.
* `pandas`: For data manipulation and reading the CSV file.
* `sklearn`: For data preprocessing (train-test split, scaling) and evaluation metrics (class weights).

## Data Preprocessing

The following steps were performed to preprocess the data:

1.  **Dropping Unnecessary Columns:** Columns like `PassengerId`, `Name`, `Ticket`, `Embarked`, and `Cabin` were removed as they were deemed less relevant for the prediction task or had too many missing values.
2.  **Converting Categorical Features:** The `Sex` column was converted to numerical values (male: 1, female: 0).
3.  **Handling Missing Values:** Rows with missing `Age` values were removed.
4.  **Splitting Data:** The data was split into training (80%) and testing (20%) sets.
5.  **Feature Scaling:** `MinMaxScaler` was used to scale the numerical features in both the training and testing sets to a range between 0 and 1. This helps in faster and more stable training of the neural network.
6.  **Handling Class Imbalance:** Class weights were calculated using `sklearn.utils.class_weight` to address the imbalance between the number of survivors and non-survivors in the training data. These weights were used during model training.

## Model Architecture

The neural network architecture consists of the following layers:

1.  A dense layer with 100 neurons, ReLU activation, and L2 regularization (with a factor of 0.001) to prevent overfitting. It takes 6 input features.
2.  A dropout layer with a rate of 0.3 to further reduce overfitting.
3.  A dense layer with 50 neurons and ReLU activation.
4.  A dropout layer with a rate of 0.2.
5.  A dense layer with 10 neurons and ReLU activation.
6.  A dropout layer with a rate of 0.1.
7.  A final dense layer with 1 neuron and sigmoid activation for binary classification (predicting the probability of survival).

## Training

The model was trained using the following settings:

* **Optimizer:** Adam
* **Loss Function:** Binary cross-entropy (suitable for binary classification)
* **Metrics:** Accuracy and Area Under the ROC Curve (AUC)
* **Validation Split:** 20% of the training data was used for validation during training.
* **Epochs:** 300
* **Batch Size:** 32
* **Callbacks:**
    * `TensorBoard`: For logging and visualization of training progress. Logs are saved in the `logs/` directory.
    * `EarlyStopping`: To stop training early if the validation loss does not improve for 5 consecutive epochs, and to restore the weights of the best epoch.
* **Class Weights:** The calculated class weights were applied during training to give more importance to the minority class (survivors).

## Evaluation

The trained model was evaluated on the test set, and the evaluation results (loss, accuracy, and AUC) are printed at the end of the script.

## How to Run

1.  Make sure you have the required libraries installed (`tensorflow`, `keras`, `numpy`, `pandas`, `scikit-learn`). You can install them using pip:
    ```bash
    pip install tensorflow numpy pandas scikit-learn
    ```
2.  Ensure that the `Titanic-Dataset.csv` file is in the same directory as the Python script.
3.  Run the Python script:
    ```bash
    python your_script_name.py
    ```
    (Replace `your_script_name.py` with the actual name of the Python file).

4.  TensorBoard logs can be viewed by running:
    ```bash
    tensorboard --logdir logs
    ```
    in your terminal and navigating to the provided URL in your web browser.

## Results

The final evaluation metrics on the test set will be printed after the training process is complete. These metrics indicate the performance of the model on unseen data.
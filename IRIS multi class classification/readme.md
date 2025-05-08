# 🌸 Iris Flower Classification with TensorFlow

This project trains a simple neural network to classify flowers in the famous **Iris dataset** into three species:

- Setosa  
- Versicolor  
- Virginica  

Using TensorFlow and a fully connected model, we achieve high accuracy on this classic multi-class classification problem.

---

## 🚀 How It Works

We use the following steps:

1. Load the **Iris dataset** using `sklearn`.
2. Standardize the features using `StandardScaler`.
3. Split the data into training and testing sets.
4. Build a 3-layer neural network with TensorFlow/Keras.
5. Train and evaluate the model.
6. Print the final accuracy.

---

## 🧠 Model Architecture

```text
Input: 4 features (sepal & petal width/length)

→ Dense(16, ReLU)  
→ Dense(8, ReLU)  
→ Dense(3, Softmax)

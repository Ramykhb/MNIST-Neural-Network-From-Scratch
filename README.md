# MNIST Neural Network From Scratch (NumPy and Pandas Only)

This repository contains a **fully implemented neural network from scratch** for handwritten digit classification on the **MNIST dataset**, using only **NumPy** and basic Python.  
No deep learning frameworks (TensorFlow, PyTorch, Keras) were used.

The project focuses on understanding the **mathematical foundations** of neural networks, including forward propagation, backpropagation, and gradient descent.

---

## 🧠 Model Architecture

- **Input Layer:**  
  - 784 features (28×28 pixels)

- **Hidden Layer 1:**  
  - 128 units  
  - ReLU activation

- **Hidden Layer 2:**  
  - 64 units  
  - ReLU activation

- **Output Layer:**  
  - 10 units  
  - Softmax activation (digit classes 0–9)

---

## ⚙️ Training Details

- **Loss Function:**  
  - Categorical Cross-Entropy

- **Optimization:**  
  - Gradient Descent (manual parameter updates)

- **Learning Rate:**  
  - Configurable (default 0.001)

- **Training Method:**  
  - Full forward propagation
  - Analytical backpropagation
  - Parameter updates using gradients

---

## 📈 Results

- **Training Accuracy:** ~**93%**
- **Loss:** Decreases smoothly during training
- **Behavior:** Stable convergence without exploding or vanishing gradients

This confirms the correctness of:
- Gradient computations
- Backpropagation logic
- Parameter updates

---

## 🧮 Implemented From Scratch

The following components are implemented **purely using math and NumPy**:

- Weight & bias initialization
- Forward propagation
- ReLU activation and derivative
- Softmax activation (numerically stable)
- Cross-entropy loss
- Backpropagation
- Gradient descent
- Accuracy calculation

---

## 🚫 No High-Level Libraries

The following libraries were **not used**:
- TensorFlow
- PyTorch
- Keras
- Scikit-learn (for modeling)

Only NumPy, Pandas and basic plotting utilities are used.

---

## 🎯 Purpose

This project was built to:
- Deeply understand how neural networks work internally
- Practice implementing ML algorithms without abstractions
- Demonstrate strong fundamentals in machine learning

---

## 🧑‍💻 Author

**Ramy Khachab**

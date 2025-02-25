
print(
"""
### 📌 Mathematics of Gradient Descent & Backpropagation for XOR MLP

#### **1️⃣ Forward Propagation Equations**

1. **Hidden Layer Calculations** - Compute weighted sum of inputs:
      ```
      H = X ⋅ W_input_hidden + B_hidden
      ```
    - Apply activation function:
      ```
      H_activated = f(H)
      ```

2. **Output Layer Calculations** - Compute weighted sum of hidden layer outputs:
      ```
      O = H_activated ⋅ W_hidden_output + B_output
      ```
    - Apply activation function (Sigmoid or Softmax):
      ```
      O_activated = g(O)
      ```

---

#### **2️⃣ Compute Error (Loss Function)**

1. **Mean Absolute Error (MAE):** ```
    Loss = (1/N) Σ |y - O_activated|
    ```
    or **Mean Squared Error (MSE):** ```
    Loss = (1/N) Σ (y - O_activated)²
    ```

---

#### **3️⃣ Backpropagation: Compute Gradients**

1. **Compute Error at Output Layer** ```
    Error = y - O_activated
    ```

2. **Compute Gradient for Output Layer (If Using Sigmoid)** - Derivative of Sigmoid activation:
      ```
      g'(O) = O_activated ⋅ (1 - O_activated)
      ```
    - Gradient for output layer:
      ```
      δ_output = Error ⋅ g'(O)
      ```

3. **Compute Gradient for Hidden Layer** - Backpropagate Error to Hidden Layer:
      ```
      Hidden Error = δ_output ⋅ W_hidden_outputᵀ
      ```
    - Compute Gradient for Hidden Layer:
      ```
      δ_hidden = Hidden Error ⋅ f'(H_activated)
      ```

---

#### **4️⃣ Update Weights & Biases Using Gradient Descent**

1. **Update Weights (Gradient Descent Rule)** - Hidden → Output Layer:
      ```
      W_hidden_output = W_hidden_output + η ⋅ H_activatedᵀ ⋅ δ_output
      ```
    - Input → Hidden Layer:
      ```
      W_input_hidden = W_input_hidden + η ⋅ Xᵀ ⋅ δ_hidden
      ```

2. **Update Biases** - Output Bias:
      ```
      B_output = B_output + η ⋅ Σ δ_output
      ```
    - Hidden Bias:
      ```
      B_hidden = B_hidden + η ⋅ Σ δ_hidden
      ```

where:
- W = Weight matrices
- B = Biases
- η = Learning rate
- X = Input data
- H = Hidden layer activations
- O = Output layer activations
- δ = Gradients for weight updates

---

### **🚀 Summary**

✅ **Forward propagation** computes activations using weights and biases. 
✅ **Loss function** measures the difference between predictions and actual values. 
✅ **Backpropagation** computes gradients using the chain rule. 
✅ **Gradient descent** updates weights and biases to minimize error.
"""
)


print(
    
    """
### 📌 Mathematics of Gradient Descent & Backpropagation for XOR MLP

#### **1️⃣ Forward Propagation Equations**

1. **Hidden Layer Calculations**  
   - Compute weighted sum of inputs:  
     H = X ⋅ W_input-hidden + B_hidden
   - Apply activation function:  
     H_activated = f(H)

2. **Output Layer Calculations**  
   - Compute weighted sum of hidden layer outputs:  
     O = H_activated ⋅ W_hidden-output + B_output
   - Apply activation function (Sigmoid or Softmax):  
     O_activated = g(O)

---

#### **2️⃣ Compute Error (Loss Function)**

1. **Mean Absolute Error (MAE):**  
   Loss = (1/N) Σ |y - O_activated|
   or **Mean Squared Error (MSE):**  
   Loss = (1/N) Σ (y - O_activated)²

---

#### **3️⃣ Backpropagation: Compute Gradients**

1. **Compute Error at Output Layer**  
   Error = y - O_activated

2. **Compute Gradient for Output Layer (If Using Sigmoid)**  
   - Derivative of Sigmoid activation:  
     g'(O) = O_activated ⋅ (1 - O_activated)
   - Gradient for output layer:  
     δ_output = Error ⋅ g'(O)

3. **Compute Gradient for Hidden Layer**  
   - Backpropagate Error to Hidden Layer:  
     Hidden Error = δ_output ⋅ W_hidden-outputᵀ
   - Compute Gradient for Hidden Layer:  
     δ_hidden = Hidden Error ⋅ f'(H_activated)

---

#### **4️⃣ Update Weights & Biases Using Gradient Descent**

1. **Update Weights (Gradient Descent Rule)**  
   - Hidden → Output Layer:  
     W_hidden-output = W_hidden-output + η ⋅ H_activatedᵀ ⋅ δ_output
   - Input → Hidden Layer:  
     W_input-hidden = W_input-hidden + η ⋅ Xᵀ ⋅ δ_hidden

2. **Update Biases**  
   - Output Bias:  
     B_output = B_output + η ⋅ Σ δ_output
   - Hidden Bias:  
     B_hidden = B_hidden + η ⋅ Σ δ_hidden

where:
- W = Weight matrices
- B = Biases
- η = Learning rate
- X = Input data
- H = Hidden layer activations
- O = Output layer activations
- δ = Gradients for weight updates

---

### **🚀 Summary**

✅ **Forward propagation** computes activations using weights and biases.  
✅ **Loss function** measures the difference between predictions and actual values.  
✅ **Backpropagation** computes gradients using the chain rule.  
✅ **Gradient descent** updates weights and biases to minimize error.
    """
)
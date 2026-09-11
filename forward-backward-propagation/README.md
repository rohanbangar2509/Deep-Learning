# Forward Propagation and Backpropagation using TensorFlow/Keras

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#forward-propagation-and-backpropagation-using-tensorflowkeras)

A practical deep learning implementation that uses a Multilayer Perceptron (MLP) built with TensorFlow/Keras to understand **forward propagation and backpropagation** using the Iris dataset.

The project also studies how different **learning rates and numbers of epochs** affect neural network training and model performance.

## 📌 Project Overview

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-project-overview)

This project demonstrates the fundamental training process of a neural network:

* Loading the Iris dataset
* Exploring the dataset
* Checking dataset information and statistics
* Checking target class distribution
* Splitting data into training and testing sets
* Standardizing input features
* Building a Multilayer Perceptron
* Compiling the neural network
* Understanding forward propagation
* Calculating loss
* Understanding backpropagation
* Updating model weights
* Training the neural network
* Evaluating model performance
* Generating predictions
* Creating a confusion matrix
* Generating a classification report
* Studying the effect of different learning rates
* Studying the effect of different numbers of epochs

The notebook focuses on understanding how a neural network learns through **forward propagation, loss calculation, backpropagation, and weight updates**.

## 🧠 Model Architecture

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-model-architecture)

The project uses a simple Multilayer Perceptron with two hidden layers.

```text
Input Features
     │
     │ 4 Features
     ▼
Dense Layer
     │
     │ 16 Neurons
     │ ReLU
     ▼
Dense Layer
     │
     │ 8 Neurons
     │ ReLU
     ▼
Output Layer
     │
     │ 3 Neurons
     │ Softmax
     ▼
Predicted Iris Class
     │
     ├── Class 0
     ├── Class 1
     └── Class 2
```

The neural network is implemented using:

```python
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])
```

## 🔄 Forward Propagation and Backpropagation

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-forward-propagation-and-backpropagation)

During neural network training, the following process takes place:

```text
Input Data
    │
    ▼
Forward Propagation
    │
    ▼
Predictions
    │
    ▼
Loss Calculation
    │
    ▼
Backpropagation
    │
    ▼
Gradient Calculation
    │
    ▼
Weight Update
    │
    ▼
Next Training Iteration
```

### Forward Propagation

Forward propagation passes the input features through the neural network layers to generate predictions.

```text
Input → Hidden Layer 1 → Hidden Layer 2 → Output → Prediction
```

### Backpropagation

Backpropagation uses the calculated loss to determine how the model weights contributed to the prediction error.

The gradients are propagated backward through the network and the optimizer updates the weights to reduce the loss.

The notebook uses the **Adam optimizer** for weight updates.

## 📊 Dataset

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-dataset)

The project uses the **Iris dataset** provided by Scikit-learn.

The dataset contains:

* 150 samples
* 4 input features
* 3 target classes

The four features are:

```text
Sepal Length
Sepal Width
Petal Length
Petal Width
```

The target classes are represented as:

```text
0
1
2
```

The dataset contains 50 samples for each class, making the target distribution balanced.

The dataset is loaded using:

```python
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
```

The notebook confirms:

```text
Feature Shape : (150, 4)
Target Shape : (150,)
```

## ⚙️ Data Preprocessing

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#%EF%B8%8F-data-preprocessing)

### Train-Test Split

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

This produces:

```text
Training Samples → 120
Testing Samples  → 30
```

### Feature Scaling

The input features are standardized using `StandardScaler`.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
```

Feature scaling is important because neural networks generally learn more effectively when input features have comparable scales.

## ⚙️ Model Compilation

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#%EF%B8%8F-model-compilation)

The model is compiled using:

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

The main components are:

* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy
* **Evaluation Metric:** Accuracy

## 🛠️ Technologies Used

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#%EF%B8%8F-technologies-used)

* Python 3.10
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

## 🚀 Getting Started

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-getting-started)

### 1. Clone the repository

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#1-clone-the-repository)

```bash
git clone https://github.com/rohanbangar2509/Deep-Learning
```

### 2. Navigate to the project directory

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#2-navigate-to-the-project-directory)

```bash
cd forward-backward-propagation
```

### 3. Create a virtual environment

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#3-create-a-virtual-environment)

```bash
python -m venv venv
```

### 4. Activate the virtual environment

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#4-activate-the-virtual-environment)

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### 5. Install dependencies

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#5-install-dependencies)

```bash
pip install -r requirements.txt
```

### 6. Launch Jupyter Notebook

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#6-launch-jupyter-notebook)

```bash
jupyter notebook
```

Open:

```text
notebooks/forward_backward_propogation.ipynb
```

Run the notebook from top to bottom.

## 📈 Results

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-results)

The main MLP model was trained for **50 epochs** using the Adam optimizer.

The final evaluation on the test dataset produced:

```text
Test Accuracy: 76.67%
Test Loss: 0.5423
```

The confusion matrix was:

```text
[[10, 0, 0],
 [ 0, 8, 1],
 [ 0, 6, 5]]
```

The classification report showed:

```text
Class 0:
Precision: 1.00
Recall:    1.00
F1-score: 1.00

Class 1:
Precision: 0.57
Recall:    0.89
F1-score: 0.70

Class 2:
Precision: 0.83
Recall:    0.45
F1-score: 0.59
```

The notebook also visualizes:

* Training accuracy
* Validation accuracy
* Training loss
* Validation loss
* Confusion matrix
* Classification performance

## 🎯 Effect of Different Learning Rates

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-effect-of-different-learning-rates)

The notebook compares four learning rates:

```text
0.1
0.01
0.001
0.0001
```

The observed results were:

```text
Learning Rate: 0.1     Accuracy: 1.0000
Learning Rate: 0.01    Accuracy: 1.0000
Learning Rate: 0.001   Accuracy: 0.9000
Learning Rate: 0.0001  Accuracy: 0.3667
```

### Observations

* **0.1:** May converge quickly but can overshoot the optimum and become unstable.
* **0.01:** Often learns faster while remaining stable.
* **0.001:** A common default learning rate for Adam and generally provides stable learning.
* **0.0001:** Very stable but learns slowly and may require more epochs.

This experiment demonstrates that the learning rate has a significant effect on the speed and quality of neural network training.

## ⏱️ Effect of Different Epochs

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-effect-of-different-epochs)

The notebook also studies the effect of different numbers of epochs.

The tested values were:

```text
10 Epochs
30 Epochs
50 Epochs
100 Epochs
```

The observed results were:

```text
Epochs: 10   Accuracy: 0.8667
Epochs: 30   Accuracy: 0.8333
Epochs: 50   Accuracy: 0.8667
Epochs: 100  Accuracy: 0.9667
```

### Observations

* **10 epochs:** The model may not have learned enough, resulting in lower accuracy.
* **30–50 epochs:** The model continues learning feature representations.
* **100 epochs:** The experiment achieved higher accuracy, although excessive training can potentially lead to overfitting if validation performance stops improving.

The experiment demonstrates that the number of epochs can significantly influence model performance.

## 🔍 Project Workflow

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-project-workflow)

```text
Iris Dataset
      │
      ▼
Load Dataset
      │
      ▼
Explore Dataset
      │
      ▼
Check Classes & Statistics
      │
      ▼
Train-Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Build MLP
      │
      ▼
Compile Model
      │
      ▼
Forward Propagation
      │
      ▼
Loss Calculation
      │
      ▼
Backpropagation
      │
      ▼
Weight Update
      │
      ▼
Train Model
      │
      ▼
Evaluate Model
      │
      ▼
Predictions
      │
      ▼
Confusion Matrix
      │
      ▼
Classification Report
      │
      ▼
Learning Rate Experiment
      │
      ▼
Epoch Experiment
```

## 📁 Repository Structure

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-repository-structure)

```text
forward-backward-propagation/
│
├── notebooks/
│   └── forward_backward_propogation.ipynb
│
├── models/
│   └── README.md
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 📚 Key Concepts Demonstrated

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-key-concepts-demonstrated)

This project demonstrates the following deep learning concepts:

* Multilayer Perceptron
* Neural network fundamentals
* Forward propagation
* Loss calculation
* Backpropagation
* Gradient-based learning
* Weight updates
* Dense layers
* ReLU activation
* Softmax activation
* Adam optimizer
* Sparse categorical cross-entropy
* Feature scaling
* Model training
* Model evaluation
* Classification accuracy
* Confusion matrix
* Precision
* Recall
* F1-score
* Learning rate
* Epochs
* Underfitting
* Overfitting
* Training and validation curves

## 🔮 Future Improvements

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-future-improvements)

Possible improvements include:

* Visualize forward propagation mathematically
* Implement forward propagation manually using NumPy
* Implement backpropagation manually using NumPy
* Compare manual backpropagation with TensorFlow/Keras
* Experiment with additional hidden layers
* Experiment with different numbers of neurons
* Compare different activation functions
* Compare different optimizers
* Perform systematic learning-rate tuning
* Add early stopping
* Add a validation split
* Visualize decision boundaries
* Perform hyperparameter tuning
* Compare the MLP with traditional machine learning classifiers

## 👨‍💻 Author

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/forward-backward-propagation#-author)

**Rohan Bangar**

B.Tech — Artificial Intelligence

---

⭐ If you found this project useful, consider giving the repository a star.

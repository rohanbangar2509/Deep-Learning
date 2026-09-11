# MLP Wine Classification using Scikit-learn

A deep learning project that uses Multi-Layer Perceptron (MLP) classifiers built with Scikit-learn to classify wines into three different categories based on their chemical properties.

## 📌 Project Overview

This project demonstrates the workflow of implementing and evaluating a Multi-Layer Perceptron for multi-class classification:

- Loading the Wine dataset
- Exploring the dataset
- Understanding dataset features and target classes
- Splitting the dataset into training and testing sets
- Scaling features using StandardScaler
- Building MLP classifiers with different architectures
- Training the models
- Generating predictions
- Evaluating model performance
- Calculating Accuracy, Precision, Recall, and F1 Score
- Generating classification reports
- Visualizing confusion matrices
- Analyzing loss curves
- Comparing the performance of different MLP architectures

The experiment compares two MLP architectures using the same 70/30 train-test split.

## 🧠 MLP Model Architectures

### MLP Model 1

The first model uses two hidden layers with 100 neurons each.

```text
Input Features
      │
      │ 13 features
      ▼
Hidden Layer 1
      │
      │ 100 neurons
      │ ReLU activation
      ▼
Hidden Layer 2
      │
      │ 100 neurons
      │ ReLU activation
      ▼
Output Layer
      │
      │ 3 classes
      ▼
Predicted Wine Class
```

Model configuration:

- Hidden Layers: `(100, 100)`
- Maximum Iterations: `2000`
- Initial Learning Rate: `0.001`
- Random State: `42`

### MLP Model 2

The second model uses a simpler architecture with a single hidden layer containing 50 neurons.

```text
Input Features
      │
      │ 13 features
      ▼
Hidden Layer
      │
      │ 50 neurons
      │ ReLU activation
      ▼
Output Layer
      │
      │ 3 classes
      ▼
Predicted Wine Class
```

Model configuration:

- Hidden Layers: `(50,)`
- Maximum Iterations: `2000`
- Initial Learning Rate: `0.001`
- Random State: `42`

The second architecture is designed to reduce network complexity while still learning the patterns present in the dataset.

## 📊 Dataset

The project uses the **Wine dataset** provided by Scikit-learn.

The dataset contains:

- **178 samples**
- **13 numerical features**
- **3 target classes**

The features represent different chemical properties of wine, including:

- Alcohol
- Malic Acid
- Ash
- Alcalinity of Ash
- Magnesium
- Total Phenols
- Flavanoids
- Nonflavanoid Phenols
- Proanthocyanins
- Color Intensity
- Hue
- OD280/OD315 of Diluted Wines
- Proline

The dataset is loaded directly using:

```python
from sklearn.datasets import load_wine

wine = load_wine()
```

No external dataset download is required.

## ⚙️ Data Preprocessing

### Train-Test Split

The dataset is divided into:

```text
Training Data : 70%
Testing Data  : 30%
```

The train-test split uses:

```python
train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)
```

### Feature Scaling

Feature scaling is performed using `StandardScaler`.

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Scaling is applied so that the numerical input features are placed on comparable scales before training the MLP models.

## 📈 Model Evaluation

The trained models are evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix

Loss curves are also visualized to analyze the training behavior of the MLP models.

## 🏆 Results

The experiment produced the following results:

| Metric | MLP Model 1 | MLP Model 2 |
|---|---:|---:|
| Accuracy | 100.00% | 98.15% |
| Precision | 100.00% | 98.27% |
| Recall | 100.00% | 98.15% |
| F1 Score | 100.00% | 98.16% |

MLP Model 1 achieved the highest performance in this experiment, achieving **100% accuracy, precision, recall, and F1 score** on the test set.

MLP Model 2 also achieved excellent performance with approximately **98.15% test accuracy**.

The notebook additionally provides confusion matrices and loss curves for both models.

## 🔍 Project Workflow

```text
Wine Dataset
      │
      ▼
Load Dataset
      │
      ▼
Explore Dataset
      │
      ▼
Split Dataset
      │
      ▼
70% Training / 30% Testing
      │
      ▼
Feature Scaling
(StandardScaler)
      │
      ▼
Build MLP Models
      │
      ├───────────────┐
      ▼               ▼
MLP Model 1       MLP Model 2
(100, 100)          (50,)
      │               │
      └───────┬───────┘
              ▼
      Generate Predictions
              │
              ▼
       Model Evaluation
              │
              ▼
 Accuracy / Precision
 Recall / F1 Score
              │
              ▼
      Confusion Matrix
              │
              ▼
        Loss Curves
              │
              ▼
      Model Comparison
```

## 🛠️ Technologies Used

- Python 3.10
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rohanbangar2509/Deep-Learning.git
```

### 2. Navigate to the project directory

```bash
cd Deep-Learning/mlp-wine-classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
notebooks/MLP_Wine.ipynb
```

Run the notebook cells sequentially from top to bottom.

## 📁 Repository Structure

```text
mlp-wine-classification/
│
├── notebooks/
│   └── MLP_Wine.ipynb
│
└── README.md
```

## 💡 Key Concepts Demonstrated

This project demonstrates the following machine learning and deep learning concepts:

- Multi-Layer Perceptron (MLP)
- Multi-class classification
- Train-test splitting
- Feature scaling
- StandardScaler
- Neural network architecture
- Hidden layers
- ReLU activation
- Adam optimization
- Model training
- Model prediction
- Classification metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Classification Report
- Confusion Matrix
- Loss Curve
- Model comparison

## 🔮 Future Improvements

Possible improvements include:

- Experiment with different train-test split ratios
- Compare additional MLP architectures
- Experiment with different learning rates
- Experiment with different numbers of iterations
- Compare different activation functions
- Compare different optimization algorithms
- Perform hyperparameter tuning
- Use cross-validation for more robust evaluation
- Analyze feature importance using additional techniques
- Compare MLP performance with other classification algorithms
- Compare the MLP with deeper neural network architectures

## 📚 Learning Outcome

Through this experiment, the following concepts were explored:

- Understanding the structure of the Wine dataset
- Preparing numerical data for neural network models
- Applying feature scaling before MLP training
- Designing MLP architectures with different complexities
- Training and evaluating neural network classifiers
- Interpreting classification metrics
- Understanding confusion matrices
- Analyzing model loss curves
- Comparing neural network architectures based on performance

## 👨‍💻 Author

**Rohan Bangar**

B.Tech — Artificial Intelligence

---

⭐ If you found this project useful, consider giving the repository a star.
# 🚦 CNN-Based Traffic Sign Classification

A Deep Learning project that uses a **Convolutional Neural Network (CNN)** with **TensorFlow/Keras** to classify traffic sign images into **43 different classes** using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The model was developed and trained using **WSL2 Ubuntu 24.04** with an **NVIDIA GeForce RTX 4050 Laptop GPU**.

---

## 📌 Project Overview

This project demonstrates an end-to-end image classification workflow, from dataset preprocessing to model evaluation and prediction.

### 🔄 Workflow

```text
GTSRB Dataset
      ↓
Image Preprocessing
      ↓
Data Augmentation
      ↓
CNN Model
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Traffic Sign Classification
```

### Key Steps

* Dataset loading and exploration
* Image preprocessing and normalization
* Train/Validation/Test splitting
* Data augmentation
* CNN model development
* Model training
* Model evaluation
* Classification report
* Confusion matrix analysis
* Misclassified image analysis
* Individual image prediction
* Model saving

---

## 📊 Dataset

**Dataset:** German Traffic Sign Recognition Benchmark (GTSRB)

| Property          |   Value |
| ----------------- | ------: |
| Total Images      |  39,209 |
| Number of Classes |      43 |
| Image Size        | 32 × 32 |
| Channels          |     RGB |
| Training Images   |  27,446 |
| Validation Images |   5,881 |
| Test Images       |   5,882 |

The images are normalized from pixel values **0–255** to **0–1** before being provided to the CNN.

The dataset contains various traffic signs, including:

* Speed limit signs
* Stop
* Yield
* No entry
* Road work
* Pedestrian crossing
* Roundabout
* Priority road
* No overtaking
* Other regulatory and warning signs

> **Note:** The GTSRB dataset is not included in this repository because it contains thousands of image files.

---

## 🧠 CNN Architecture

The model consists of **three convolutional blocks**, followed by fully connected layers for classification.

```text
Input (32 × 32 × 3)
        ↓
Data Augmentation
        ↓
Conv2D (32 filters) + ReLU
        ↓
MaxPooling
        ↓
Conv2D (64 filters) + ReLU
        ↓
MaxPooling
        ↓
Conv2D (128 filters) + ReLU
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense (128) + ReLU
        ↓
Dropout (0.5)
        ↓
Dense (43) + Softmax
```

**Total Parameters:** 361,067

### Data Augmentation

The training pipeline applies:

* Random rotation
* Random zoom
* Random translation

### Training Configuration

| Parameter         | Configuration                    |
| ----------------- | -------------------------------- |
| Optimizer         | Adam                             |
| Loss Function     | Sparse Categorical Cross-Entropy |
| Batch Size        | 64                               |
| Maximum Epochs    | 30                               |
| Early Stopping    | Enabled                          |
| ReduceLROnPlateau | Enabled                          |

---

## 📈 Results

The trained CNN achieved strong performance on the test dataset.

### Model Performance

| Metric                   |      Score |
| ------------------------ | ---------: |
| Training Accuracy        | **96.62%** |
| Validation Accuracy      | **99.12%** |
| Test Accuracy            | **99.27%** |
| Test Loss                | **0.0254** |
| Best Validation Accuracy | **99.15%** |
| Best Validation Epoch    |     **26** |

### Classification Report

| Metric            |      Score |
| ----------------- | ---------: |
| Accuracy          | **99.27%** |
| Macro Precision   | **99.28%** |
| Macro Recall      | **99.45%** |
| Macro F1-Score    | **99.36%** |
| Weighted F1-Score | **99.27%** |

A **43 × 43 confusion matrix** was generated to analyze class-level performance.

Misclassified test images were also examined to understand where the model made incorrect predictions.

---

## 🔍 Model Evaluation

The notebook includes several evaluation techniques:

### 📊 Confusion Matrix

A **43 × 43 confusion matrix** is used to analyze predictions across all traffic sign classes and identify classes that are frequently confused with one another.

### ❌ Misclassified Images

Incorrectly classified test images are visualized to understand:

* Which traffic signs are difficult to classify
* Which classes are commonly confused
* Whether image quality affects predictions
* Potential limitations of the trained CNN

### 🖼️ Individual Image Prediction

The trained model can also be used to classify individual traffic sign images.

---

## 💾 Trained Model

The trained CNN model is saved in **Keras format**:

```text
models/
└── traffic_sign_cnn.keras
```

The saved model can be loaded later to make predictions without retraining the network.

---

## 🛠️ Technologies Used

### Programming & Deep Learning

* Python
* TensorFlow
* Keras

### Data Processing

* NumPy
* Pandas
* Scikit-learn

### Image Processing

* OpenCV
* Pillow

### Visualization

* Matplotlib
* Seaborn

### Development Environment

* Jupyter Notebook
* WSL2
* Ubuntu 24.04
* CUDA
* cuDNN
* NVIDIA GeForce RTX 4050 Laptop GPU

---

## 📁 Project Structure

```text
traffic-sign-cnn/
│
├── data/
│   └── Train/
│       ├── 0/
│       ├── 1/
│       ├── 2/
│       ├── ...
│       └── 42/
│
├── models/
│   └── traffic_sign_cnn.keras
│
├── notebooks/
│   └── CNN_Traffic_Sign_Classification.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd traffic-sign-cnn
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

Upgrade `pip`:

```bash
pip install --upgrade pip
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Download the **GTSRB training dataset** and place it inside:

```text
data/Train/
```

The directory should contain the 43 class folders:

```text
data/Train/
├── 0/
├── 1/
├── 2/
├── ...
└── 42/
```

### 5. Run the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/CNN_Traffic_Sign_Classification.ipynb
```

Run the notebook **from top to bottom** to reproduce the preprocessing, training, evaluation, and prediction workflow.

---

## 🎯 Key Learning Outcomes

Through this project, I explored and implemented:

* Image classification using CNNs
* Image preprocessing and normalization
* Data augmentation
* Train/validation/test splitting
* CNN architecture design
* Model training and validation
* Overfitting control
* Dropout regularization
* Early stopping
* Learning-rate scheduling
* Classification metrics
* Confusion matrix analysis
* Misclassified image analysis
* Individual image prediction
* Model saving and loading

---

## 📌 Project Highlights

* 🧠 Built a CNN for **43-class traffic sign classification**
* 🖼️ Worked with **39,209 traffic sign images**
* ⚡ Trained using an **NVIDIA RTX 4050 Laptop GPU**
* 📈 Achieved **99.27% test accuracy**
* 📊 Performed detailed confusion matrix analysis
* 🔍 Analyzed misclassified test images
* 💾 Saved the trained model in `.keras` format
* 🔄 Implemented an end-to-end Deep Learning workflow

---

## 👨‍💻 Author

**Rohan Bangar**

B.Tech – Computer Science & Engineering (Artificial Intelligence)

---

⭐ If you found this project useful, feel free to explore the notebook for the complete implementation and analysis.

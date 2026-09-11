# 🚦 CNN-Based Traffic Sign Classification

A Deep Learning project that uses a **Convolutional Neural Network (CNN)** with **TensorFlow/Keras** to classify traffic sign images into **43 different classes** using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset.

The model was developed and trained using **WSL2 Ubuntu 24.04** with an **NVIDIA GeForce RTX 4050 Laptop GPU**.

---

## 📌 Project Overview

This project demonstrates an end-to-end image classification workflow:

- Dataset loading and exploration
- Image preprocessing
- Train/Validation/Test splitting
- Data augmentation
- CNN model development
- Model training
- Model evaluation
- Confusion matrix analysis
- Misclassified image analysis
- Individual image prediction
- Model saving

### Workflow

```text
GTSRB Dataset
      ↓
Image Preprocessing
      ↓
Data Augmentation
      ↓
CNN
      ↓
Classification
      ↓
Traffic Sign Class
📊 Dataset

Dataset: German Traffic Sign Recognition Benchmark (GTSRB)

Property	Value
Total Images	39,209
Classes	43
Image Size	32 × 32
Channels	RGB
Training Images	27,446
Validation Images	5,881
Test Images	5,882

The images are normalized from pixel values 0–255 to 0–1.

The dataset contains traffic signs such as speed limits, stop, yield, no entry, road work, pedestrian crossing, roundabout, and many others.

🧠 CNN Architecture

The model consists of three convolutional blocks followed by fully connected layers.

Input (32×32×3)
      ↓
Data Augmentation
      ↓
Conv2D (32) + ReLU
      ↓
MaxPooling
      ↓
Conv2D (64) + ReLU
      ↓
MaxPooling
      ↓
Conv2D (128) + ReLU
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

Total Parameters: 361,067

Data Augmentation
Random Rotation
Random Zoom
Random Translation
Training
Optimizer: Adam
Loss: Sparse Categorical Cross-Entropy
Batch Size: 64
Maximum Epochs: 30
EarlyStopping
ReduceLROnPlateau
📈 Results

The trained CNN achieved strong performance on the test dataset.

Metric	Score
Training Accuracy	96.62%
Validation Accuracy	99.12%
Test Accuracy	99.27%
Test Loss	0.0254

Best Validation Accuracy: 99.15% at Epoch 26

Classification Report
Metric	Score
Accuracy	99.27%
Macro Precision	99.28%
Macro Recall	99.45%
Macro F1-Score	99.36%
Weighted F1-Score	99.27%

A 43 × 43 confusion matrix and misclassified test images are also analyzed in the notebook.

🛠️ Technologies Used
Python
TensorFlow / Keras
NumPy
Pandas
Scikit-learn
OpenCV / Pillow
Matplotlib
Seaborn
Jupyter Notebook
WSL2 Ubuntu
CUDA / cuDNN
💾 Trained Model

The trained model is saved in Keras format:

models/
└── traffic_sign_cnn.keras

The saved model can be loaded later for making predictions without retraining.

📁 Project Structure
traffic-sign-cnn/
│
├── data/
│   └── Train/
│       ├── 0/
│       ├── 1/
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

The GTSRB dataset is not included in the repository because it contains thousands of image files.

🚀 Getting Started
1. Clone the Repository
git clone <YOUR_REPOSITORY_URL>
cd traffic-sign-cnn
2. Create Virtual Environment
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
4. Add Dataset

Place the GTSRB training dataset inside:

data/Train/

with folders:

0/
1/
2/
...
42/
5. Run the Notebook
jupyter notebook

Open:

notebooks/CNN_Traffic_Sign_Classification.ipynb

and run the notebook from top to bottom.

🎯 Key Learning Outcomes

Through this project, I explored:

Image classification using CNNs
Data preprocessing and normalization
Data augmentation
Model training and validation
Overfitting control
Learning-rate scheduling
Classification metrics
Confusion matrix analysis
Model prediction and saving


👨‍💻 Author

Rohan Bangar

B.Tech – Computer Science & Engineering (Artificial Intelligence)



**This version is much better for GitHub**: it gives recruiters/visitors the important information quickly, while your `.ipynb` contains the detailed implementation and analysis.
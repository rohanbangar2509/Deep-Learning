# Transfer Learning using AlexNet, VGG16, ResNet50, and EfficientNetB0

A deep learning project that demonstrates transfer learning using pre-trained AlexNet, VGG16, ResNet50, and EfficientNetB0 models for image classification using the CIFAR-10 dataset.

## 📌 Project Overview

This project demonstrates the workflow of using pre-trained convolutional neural networks for image classification through transfer learning.

The project includes:

* Loading the CIFAR-10 dataset
* Exploring and preprocessing the dataset
* Resizing images to 224 × 224 pixels
* Normalizing images using ImageNet statistics
* Loading ImageNet pre-trained models
* Freezing pre-trained feature extraction layers
* Replacing the final classification layer
* Training the modified models
* Evaluating model performance
* Generating classification reports
* Generating confusion matrices
* Comparing model accuracy
* Comparing model training time
* Comparing model parameter counts

The following pre-trained models are implemented:

* AlexNet
* VGG16
* ResNet50
* EfficientNetB0

The models are used as feature extractors, while their final classification layers are replaced to classify the 10 classes of the CIFAR-10 dataset.

## 🧠 Model Architectures

The project uses four different ImageNet-pre-trained CNN architectures.

### AlexNet

```text
Input Image
    │
    │ 224 × 224 × 3
    ▼
Pre-trained AlexNet
    │
    │ Frozen Feature Layers
    ▼
Classifier
    │
    │ 10 classes
    ▼
CIFAR-10 Prediction
```

### VGG16

```text
Input Image
    │
    │ 224 × 224 × 3
    ▼
Pre-trained VGG16
    │
    │ Frozen Feature Layers
    ▼
Classifier
    │
    │ 10 classes
    ▼
CIFAR-10 Prediction
```

### ResNet50

```text
Input Image
    │
    │ 224 × 224 × 3
    ▼
Pre-trained ResNet50
    │
    │ Frozen Feature Layers
    ▼
Fully Connected Layer
    │
    │ 10 classes
    ▼
CIFAR-10 Prediction
```

### EfficientNetB0

```text
Input Image
    │
    │ 224 × 224 × 3
    ▼
Pre-trained EfficientNetB0
    │
    │ Frozen Feature Layers
    ▼
Classifier
    │
    │ 10 classes
    ▼
CIFAR-10 Prediction
```

## 📊 Dataset

The project uses the **CIFAR-10 dataset**, which contains 60,000 color images belonging to 10 different classes.

The images originally have dimensions:

```text
32 × 32 × 3
```

The 10 classes are:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

The dataset is divided into:

```text
Training Set:  50,000 images
Test Set:      10,000 images
```

The training set is further divided into:

```text
Training Set:    40,000 images
Validation Set:  10,000 images
```

The dataset is loaded using:

```python
torchvision.datasets.CIFAR10
```

## ⚙️ Data Preprocessing

Since the pre-trained models were originally trained on ImageNet, the CIFAR-10 images are resized to:

```text
224 × 224 pixels
```

The images are converted to tensors and normalized using ImageNet mean and standard deviation values.

```python
Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

The preprocessing pipeline is:

```text
CIFAR-10 Image
      │
      ▼
Resize to 224 × 224
      │
      ▼
Convert to Tensor
      │
      ▼
ImageNet Normalization
      │
      ▼
Model Input
```

## 🔄 Transfer Learning Approach

Transfer learning allows a model trained on a large dataset such as ImageNet to be reused for another image classification task.

The general approach used in this project is:

```text
Pre-trained ImageNet Model
          │
          ▼
Freeze Feature Extraction Layers
          │
          ▼
Replace Final Classification Layer
          │
          ▼
Train New Classification Layer
          │
          ▼
CIFAR-10 Classification
```

The pre-trained feature extraction layers are frozen, while the final classification layer is modified to produce predictions for the 10 CIFAR-10 classes.

## 🛠️ Technologies Used

* Python 3.10+
* PyTorch
* TorchVision
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rohanbangar2509/Deep-Learning.git
```

### 2. Navigate to the project directory

```bash
cd Deep-Learning
```

Navigate to the transfer learning directory if applicable:

```bash
cd transfer-learning
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
pip install torch torchvision matplotlib pandas numpy scikit-learn seaborn
```

If a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

### 6. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the notebooks from the project directory and run them from top to bottom.

## 📓 Notebooks

The project contains separate notebooks for each pre-trained architecture.

### 01 — AlexNet

```text
01_AlexNet_Transfer_Learning_Professional.ipynb
```

Implements transfer learning using the ImageNet-pre-trained AlexNet model.

### 02 — VGG16

```text
02_VGG16_Transfer_Learning_Professional.ipynb
```

Implements transfer learning using the ImageNet-pre-trained VGG16 model.

### 03 — ResNet50

```text
03_ResNet50_Transfer_Learning_Professional.ipynb
```

Implements transfer learning using the ImageNet-pre-trained ResNet50 model.

### 04 — EfficientNetB0

```text
04_EfficientNetB0_Transfer_Learning_Professional.ipynb
```

Implements transfer learning using the ImageNet-pre-trained EfficientNetB0 model.

### 05 — Model Comparison

```text
05_Model_Comparison_Professional.ipynb
```

Combines the results obtained from the four models and provides a comparison of:

* Test accuracy
* Test loss
* Training time
* Total parameters
* Trainable parameters

## 📈 Results

Each model notebook evaluates the trained model on the CIFAR-10 test dataset.

The following metrics are recorded:

```text
Test Accuracy
Test Loss
Training Time
Total Parameters
Trainable Parameters
```

The individual model results are saved in the `results/` directory.

```text
results/
├── alexnet_results.csv
├── vgg16_results.csv
├── resnet50_results.csv
└── efficientnetb0_results.csv
```

The final comparison notebook reads these files and generates visual comparisons of the models.

> The actual accuracy and training-time values depend on the hardware, software environment, and training configuration used when the notebooks are executed.

## 🔍 Project Workflow

```text
CIFAR-10 Dataset
       │
       ▼
Load Dataset
       │
       ▼
Split Training and Validation Data
       │
       ▼
Resize Images to 224 × 224
       │
       ▼
Normalize Images
       │
       ▼
Load ImageNet Pre-trained Model
       │
       ▼
Freeze Feature Extraction Layers
       │
       ▼
Replace Classification Layer
       │
       ▼
Train Model
       │
       ▼
Validate Model
       │
       ▼
Evaluate on Test Dataset
       │
       ▼
Generate Classification Report
       │
       ▼
Generate Confusion Matrix
       │
       ▼
Save Model Results
       │
       ▼
Compare All Models
```

## 📁 Repository Structure

```text
transfer-learning/
│
├── 01_AlexNet_Transfer_Learning_Professional.ipynb
├── 02_VGG16_Transfer_Learning_Professional.ipynb
├── 03_ResNet50_Transfer_Learning_Professional.ipynb
├── 04_EfficientNetB0_Transfer_Learning_Professional.ipynb
├── 05_Model_Comparison_Professional.ipynb
│
├── data/
│   └── CIFAR-10 dataset
│
├── results/
│   ├── alexnet_results.csv
│   ├── vgg16_results.csv
│   ├── resnet50_results.csv
│   ├── efficientnetb0_results.csv
│   └── model_comparison.csv
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 💾 Pre-trained Models

The project uses ImageNet-pre-trained models provided by TorchVision.

The models are loaded using the TorchVision pre-trained weights API.

Examples:

```python
models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
```

```python
models.vgg16(weights=models.VGG16_Weights.DEFAULT)
```

```python
models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
```

```python
models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)
```

The pre-trained feature extraction layers are frozen during training, while the final classification layer is adapted for CIFAR-10.

## 📊 Model Comparison

The final comparison notebook generates visualizations for comparing the four architectures.

### Test Accuracy

The test accuracy of each model is compared using a bar chart.

```text
Model
  │
  ├── AlexNet
  ├── VGG16
  ├── ResNet50
  └── EfficientNetB0
          │
          ▼
   Test Accuracy
```

### Training Time

The training time of each model is compared to understand computational requirements.

### Model Parameters

The total number of parameters and trainable parameters are compared to understand model size and training complexity.

## 📚 Key Concepts Demonstrated

This project demonstrates the following deep learning concepts:

* Transfer learning
* Pre-trained CNN models
* ImageNet pre-trained weights
* Feature extraction
* Parameter freezing
* Fine-tuning concepts
* Image preprocessing
* Image resizing
* Image normalization
* CIFAR-10 image classification
* Convolutional neural networks
* Classification layers
* Cross-entropy loss
* Adam optimizer
* Model training
* Model validation
* Model evaluation
* Classification reports
* Confusion matrices
* Model performance comparison

## 🔮 Future Improvements

Possible improvements include:

* Fine-tune selected layers of each pre-trained model
* Compare different learning rates
* Increase the number of training epochs
* Apply data augmentation
* Experiment with different batch sizes
* Use learning-rate schedulers
* Compare frozen feature extraction with full fine-tuning
* Add early stopping
* Save trained model weights
* Compare GPU and CPU training performance
* Add per-class accuracy analysis
* Perform hyperparameter tuning
* Test the models on another image classification dataset

## 👨‍💻 Author

**Rohan Bangar**

B.Tech — Artificial Intelligence

---

⭐ If you found this project useful, consider giving the repository a star.

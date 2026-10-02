# Image Classification using Pretrained Vision Transformer vs CNN

A professional deep learning project that compares a **pretrained Vision Transformer (ViT-B/16)** with a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10** dataset.

## 📌 Assignment

> **Perform Image classification using Pretrained Vision Transformer model and compare its performance with CNN Model.**

This project implements the complete experimental workflow:

* Load and explore the CIFAR-10 dataset
* Preprocess and augment images
* Build a CNN baseline from scratch
* Load a pretrained Vision Transformer
* Adapt ViT for CIFAR-10 classification
* Fine-tune both models
* Evaluate both models on the same test set
* Calculate accuracy, precision, recall and F1-score
* Generate confusion matrices
* Compare training history
* Compare model size
* Visualize sample predictions
* Save experiment results
* Provide a reusable prediction script

## 🧠 Models

### CNN Baseline

A custom CNN is trained from scratch using convolution, batch normalization, ReLU, max pooling, adaptive average pooling and dropout.

```text
Input Image
    ↓
Conv Block 1
    ↓
Max Pooling
    ↓
Conv Block 2
    ↓
Max Pooling
    ↓
Conv Block 3
    ↓
Adaptive Average Pooling
    ↓
Fully Connected Layer
    ↓
10-Class Output
```

### Pretrained Vision Transformer

The project uses the ImageNet-pretrained `ViT-B/16` model from Torchvision. Its original ImageNet classification head is replaced with a 10-class CIFAR-10 head and the model is fine-tuned.

```text
Input Image
    ↓
Resize to 224 × 224
    ↓
Patch Embedding
    ↓
Transformer Encoder
    ↓
Classification Head
    ↓
10-Class Output
```

## 📊 Dataset

The project uses **CIFAR-10**, a standard image-classification benchmark containing:

```text
50,000 training images
10,000 test images
10 classes
32 × 32 RGB images
```

Classes:

```text
0 → airplane
1 → automobile
2 → bird
3 → cat
4 → deer
5 → dog
6 → frog
7 → horse
8 → ship
9 → truck
```

CIFAR-10 is downloaded automatically through Torchvision on the first run.

## ⚙️ Preprocessing

### CNN

CIFAR-10 images remain at `32 × 32` resolution. Training uses random cropping, horizontal flipping and CIFAR-10 normalization.

### ViT

CIFAR-10 images are resized/cropped to `224 × 224`, followed by ImageNet normalization, matching the preprocessing expected by the pretrained ViT.

## 🔬 Experimental Design

Both models use the same CIFAR-10 train/test subset and random seed. The CNN is trained from scratch, while ViT starts from ImageNet-pretrained weights.

Default practical configuration:

```text
Training samples : 10,000
Test samples     : 2,000
CNN epochs       : 5
ViT epochs       : 3
CNN batch size   : 64
ViT batch size   : 8
```

Set `TRAIN_SAMPLES = None` and `TEST_SAMPLES = None` in `src/train.py` or the notebook to use the complete CIFAR-10 splits.

## 🛠️ Technologies Used

* Python 3.10
* PyTorch
* Torchvision
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* tqdm
* Jupyter Notebook
* CUDA / NVIDIA GPU

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the project directory

```bash
cd ViT_vs_CNN_Image_Classification
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS / WSL

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Launch Jupyter

```bash
jupyter notebook
```

Open:

```text
notebooks/ViT_vs_CNN_CIFAR10.ipynb
```

Run the notebook sequentially from top to bottom.

## ▶️ Training Using Python Script

Run the complete experiment with:

```bash
python src/train.py
```

The script downloads CIFAR-10, trains both models, evaluates them and saves the best checkpoints and metrics.

## 🔍 Custom Prediction

After training, classify an external image with:

```bash
python src/predict.py --image path/to/image.jpg
```

Example output:

```text
Device: cuda
CNN Prediction : dog (0.8421)
ViT Prediction : dog (0.9134)
```

The actual predictions and confidence values depend on the trained checkpoints and input image.

## 📈 Results

The experiment evaluates both models using:

```text
Accuracy
Precision
Recall
F1-score
```

It also generates confusion matrices and training curves.

Record the actual measured results after running the experiment:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| CNN | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| Pretrained ViT-B/16 | XX.XX% | XX.XX% | XX.XX% | XX.XX% |

The README intentionally does not hard-code experimental results.

## 🔄 Project Workflow

```text
CIFAR-10 Dataset
       ↓
Dataset Exploration
       ↓
 ┌─────┴─────┐
 ↓           ↓
CNN         ViT
 ↓           ↓
Train       Fine-tune
 └─────┬─────┘
       ↓
Common Test Set
       ↓
Accuracy / Precision / Recall / F1
       ↓
Confusion Matrices
       ↓
Model Comparison
```

## 📁 Repository Structure

```text
ViT_vs_CNN_Image_Classification/
│
├── notebooks/
│   └── ViT_vs_CNN_CIFAR10.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── outputs/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 💾 Model Checkpoints

After training, checkpoints are created at:

```text
models/cnn/cnn_cifar10.pth
models/vit/vit_cifar10.pth
```

Large model files are excluded from Git through `.gitignore` and can be regenerated by running `python src/train.py`.

## ⚡ GPU Acceleration

The project automatically uses CUDA when available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

For a 6 GB GPU, the default ViT batch size of 8 and mixed-precision training are intended to reduce memory usage. If GPU memory is insufficient, reduce `BATCH_SIZE_VIT` to 4.

## 📊 CNN vs Vision Transformer

CNNs learn local spatial features using convolution operations and have strong image-specific inductive biases.

Vision Transformers divide images into patches and use self-attention to model relationships between patches. In this project, ViT also benefits from ImageNet pretraining before being fine-tuned on CIFAR-10.

The comparison should be based on the measured experimental results rather than assuming that either architecture will always perform better.

## 🎯 Learning Outcomes

* Image classification using deep learning
* CNN architecture
* Vision Transformer architecture
* Image patch embeddings
* Self-attention
* Transfer learning
* Fine-tuning pretrained models
* Image preprocessing and augmentation
* GPU acceleration
* Mixed-precision training
* Classification metrics
* Confusion-matrix analysis
* Model comparison

## 🔮 Future Improvements

* Train both models on the complete CIFAR-10 dataset
* Increase training epochs
* Perform hyperparameter tuning
* Compare ResNet with ViT
* Compare ViT with Swin Transformer
* Freeze and unfreeze different parts of ViT
* Add early stopping and learning-rate warm-up
* Perform detailed class-wise error analysis
* Add Grad-CAM for CNN interpretability
* Visualize ViT attention maps
* Build a Streamlit application
* Build a FastAPI inference API

## 📚 References

* PyTorch Documentation — https://pytorch.org/docs/stable/index.html
* Torchvision Documentation — https://pytorch.org/vision/stable/index.html
* CIFAR-10 Dataset — https://www.cs.toronto.edu/~kriz/cifar.html
* Vision Transformer Paper — https://arxiv.org/abs/2010.11929
* Torchvision Vision Transformer — https://pytorch.org/vision/stable/models/vision_transformer.html

## 👨‍💻 Author

**Rohan Bangar**

B.Tech — Artificial Intelligence

---

⭐ If you found this project useful, consider giving the repository a star.

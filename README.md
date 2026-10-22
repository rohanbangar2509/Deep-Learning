# 🧠 Deep Learning Lab

A hands-on collection of deep learning experiments and projects covering neural network fundamentals, computer vision, transfer learning, sequence modeling, NLP, Transformers, and audio applications.

This repository is a practical learning portfolio: each project explores a specific problem, implements a model or workflow, and documents the approach, tools, and evaluation. Project-level READMEs contain the detailed setup instructions and methodology.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Projects-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Projects-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Notebooks-Jupyter-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## 🧭 Project Index

| Project | Focus | Main tools / methods |
|---|---|---|
| [Forward & Backward Propagation](./forward-backward-propagation) | MLP fundamentals, gradients, learning-rate and epoch experiments | TensorFlow/Keras, Iris dataset |
| [MNIST Digit Classification](./mnist-digit-classification) | Handwritten digit classification | TensorFlow/Keras, MNIST |
| [MLP Wine Classification](./mlp-wine-classification) | Multi-class classification on tabular data; compare MLP architectures | scikit-learn, Wine dataset |
| [Traffic Sign CNN](./traffic-sign-cnn) | 43-class traffic-sign image classification | TensorFlow/Keras, CNN, GTSRB |
| [Transfer Learning](./Transfer-Learning) | Compare pretrained CNN architectures on image classification | AlexNet, VGG16, ResNet50, EfficientNetB0, CIFAR-10 |
| [RNN, LSTM & GRU Sequence Classification](./RNN-LSTM-GRU-Sequence-Classification) | Compare recurrent networks for text sequence classification | TensorFlow/Keras, IMDb reviews |
| [LSTM Sales Forecasting](./LSTM-Sales-Forecasting) | Next-day sales forecasting from historical observations | TensorFlow/Keras, time series |
| [BERT IMDb Sentiment Analysis](./BERT_Sentiment_Analysis_IMDb) | Fine-tune a pretrained Transformer for review sentiment | PyTorch, Hugging Face Transformers, BERT |
| [Vision Transformer vs CNN](./ViT_vs_CNN_Image_Classification) | Experimentally compare a pretrained ViT with a CNN | PyTorch, Torchvision, CIFAR-10 |
| [Audio Classification, STT & TTS](./audio_classification_stt_tts) | Spoken-command classification and speech input/output | TensorFlow/Keras, CNN, log-Mel spectrograms, STT/TTS |

---

## 📚 Topics Covered

### Neural network foundations
- Artificial neural networks and multilayer perceptrons
- Neurons, layers, weights, and biases
- Forward propagation, loss calculation, backpropagation, and gradient-based optimization
- Activation and loss functions, learning rate, epochs, and batch size
- Feature scaling, train/test splits, and classification metrics

### Computer vision
- Image loading, visualization, resizing, normalization, and augmentation
- Convolution, filters, feature maps, pooling, and dense classification heads
- Multi-class image classification and error analysis
- Transfer learning, frozen feature extractors, and fine-tuning
- CNN and Vision Transformer comparison

### Sequence modeling and forecasting
- SimpleRNN, LSTM, and GRU
- Embedding layers, hidden states, memory cells, and gates
- Text sequence classification
- Time-series windows, chronological splits, and sales forecasting

### NLP and Transformers
- Text tokenization and padding
- BERT fine-tuning for binary sentiment classification
- Attention masks, pretrained language models, and inference
- Accuracy, precision, recall, F1-score, and confusion matrices

### Audio applications
- Waveform preprocessing and log-Mel spectrograms
- CNN-based spoken-command classification
- Speech-to-text (STT) and text-to-speech (TTS) integration
- Model saving and prediction on new audio

---

## 🏆 Documented Project Highlight

### Traffic Sign Classification — GTSRB

The [Traffic Sign CNN project](./traffic-sign-cnn) documents a CNN trained to classify **43 traffic-sign categories** from the German Traffic Sign Recognition Benchmark.

| Metric | Reported result |
|---|---:|
| Test accuracy | **99.27%** |
| Macro precision | **99.28%** |
| Macro recall | **99.45%** |
| Macro F1-score | **99.36%** |

These values are taken from the project's README and apply to that documented experiment, not to every model in this repository. See the project page for its architecture, configuration, evaluation details, and reproduction steps.

---

## 🛠️ Tools & Technologies

| Area | Tools |
|---|---|
| Language | Python |
| Deep learning frameworks | TensorFlow, Keras, PyTorch, Torchvision |
| NLP / pretrained models | Hugging Face Transformers and Datasets |
| Data processing | NumPy, Pandas |
| Classical ML and metrics | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Experiment environment | Jupyter Notebook |
| Acceleration | CUDA where supported by the installed framework, driver, and hardware |
| Version control | Git and GitHub |

The projects do **not** all use one identical Python environment. Some are TensorFlow/Keras projects and others use PyTorch/Torchvision or Hugging Face libraries. Follow the requirements and setup instructions inside the specific project folder.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rohanbangar2509/Deep-Learning.git
cd Deep-Learning
```

### 2. Choose a project

Open one of the folders listed in the [Project Index](#-project-index) and read its `README.md` before installing dependencies or running code.

### 3. Create a project-specific environment

For example, from the repository root:

```bash
python -m venv .venv
```

Activate it using the command for your operating system.

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS / WSL**

```bash
source .venv/bin/activate
```

Then move into the chosen project directory and install that project's dependencies, where a project-level `requirements.txt` is provided:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Environment note:** The root `requirements.txt` is not a universal lockfile for every project. Use the dependency file and Python/framework compatibility notes for the project you intend to run. GPU support depends on the specific framework build, operating system, driver, and hardware; CPU execution may be available as a fallback.

### 4. Run the notebook or script

Use the commands and notebook paths documented in the selected project's README. Many projects are notebook-first; some also include command-line training or inference scripts.

Datasets and trained checkpoints may be downloaded or generated when a project runs. Check the project's instructions before assuming these files are already present in the repository.

---

## 📁 Repository Structure

The repository currently contains these top-level project folders:

```text
Deep-Learning/
├── BERT_Sentiment_Analysis_IMDb/
├── LSTM-Sales-Forecasting/
├── RNN-LSTM-GRU-Sequence-Classification/
├── Transfer-Learning/
├── ViT_vs_CNN_Image_Classification/
├── audio_classification_stt_tts/
├── forward-backward-propagation/
├── mlp-wine-classification/
├── mnist-digit-classification/
├── traffic-sign-cnn/
├── .gitignore
├── README.md
└── requirements.txt
```

See the individual project directories for notebooks, scripts, project-specific dependencies, dataset instructions, and outputs.

---

## 📊 Evaluation Practices

Depending on the task, projects use metrics and diagnostic artifacts such as:

- **Classification:** accuracy, precision, recall, F1-score, classification reports, and confusion matrices.
- **Forecasting:** MAE, RMSE, and plots comparing actual and predicted values.
- **Model comparison:** training history, parameter counts, training time, and common test sets where documented.
- **Error analysis:** sample predictions and misclassified examples.

Results depend on the dataset split, preprocessing, hyperparameters, random seed, framework version, and compute environment. Refer to each project's README for reported results; do not assume values are directly comparable across different tasks or datasets.

---

## 🔭 Possible Next Steps

Potential extensions across the collection include:
- Add reproducible experiment configurations and consistent metric exports.
- Compare baseline models against stronger architectures on the same held-out data.
- Track experiments, seeds, package versions, and hardware used.
- Add lightweight inference demos or APIs for selected projects.
- Improve automated tests and environment setup instructions.

---

## 👨‍💻 Author

**Rohan Bangar**  
B.Tech — Computer Science & Engineering (Artificial Intelligence)  
Vishwakarma Institute of Technology, Pune

---

⭐ If you find the repository useful, explore the individual projects and consider giving it a star.

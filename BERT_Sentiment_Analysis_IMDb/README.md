# BERT Sentiment Analysis using Hugging Face Transformers

A deep learning project that uses a pre-trained BERT (`bert-base-uncased`) model with PyTorch and Hugging Face Transformers to classify movie reviews as positive or negative using the IMDb dataset.

## 📌 Project Overview

This project demonstrates the complete workflow of applying a pre-trained Transformer model to a Natural Language Processing (NLP) classification problem.

The project covers:

* Loading the IMDb sentiment-analysis dataset
* Exploring the text dataset
* Analyzing class distribution
* Analyzing review lengths
* Loading a pre-trained BERT tokenizer
* Tokenizing text data
* Creating input IDs and attention masks
* Preparing PyTorch DataLoaders
* Loading a pre-trained BERT model
* Adding a classification head
* Fine-tuning BERT on the IMDb dataset
* Validating the model during training
* Evaluating model performance
* Calculating accuracy, precision, recall, and F1-score
* Generating a confusion matrix
* Visualizing training and validation performance
* Performing sentiment prediction on custom text
* Saving the fine-tuned BERT model

The project uses **transfer learning**, where a BERT model that has already learned general language representations from large-scale text is fine-tuned for the specific task of binary sentiment classification.

## 🧠 Model Architecture

The project uses the pre-trained `bert-base-uncased` model followed by a classification head for two sentiment classes.

```text
Input Movie Review
        │
        ▼
BERT Tokenizer
        │
        ├── Input IDs
        ├── Attention Mask
        └── Token Type IDs
        │
        ▼
Pre-trained BERT
        │
        │ Contextual Language Representation
        ▼
Classification Head
        │
        │ 2 Output Logits
        ▼
Softmax
        │
        ├── Negative
        └── Positive
```

### Model Configuration

```text
Base Model        : bert-base-uncased
Task              : Binary Text Classification
Number of Classes : 2
Optimizer         : AdamW
Learning Rate     : 2e-5
Maximum Length    : 256 tokens
Batch Size        : 8
Epochs            : 2
Weight Decay      : 0.01
```

The configuration can be modified according to available computational resources.

## 📊 Dataset

The project uses the **IMDb Movie Review Dataset**, a standard dataset for binary sentiment classification.

The dataset is available through the Hugging Face Datasets library:

```text
stanfordnlp/imdb
```

Each review is associated with one of two sentiment labels:

```text
0 → Negative
1 → Positive
```

The dataset is loaded using:

```python
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
```

For this project, a configurable subset of the dataset is used for training and experimentation.

### Dataset Configuration

```text
Training samples   : 8,000
Validation samples : 2,000
Testing samples    : 2,000
```

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

* Tokenization
* Truncation of long reviews
* Maximum sequence length control
* Creation of attention masks
* Dynamic padding during batching
* Conversion to PyTorch tensors

The tokenizer is loaded using:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased"
)
```

Reviews are tokenized using:

```python
tokenizer(
    text,
    truncation=True,
    max_length=256
)
```

Dynamic padding is performed using:

```python
DataCollatorWithPadding
```

This avoids unnecessary padding and makes batch processing more efficient.

## 🤖 Transfer Learning

This project uses **transfer learning** with a pre-trained BERT model.

Instead of training a Transformer model from scratch, the project starts with:

```text
bert-base-uncased
```

A classification head is added for the IMDb sentiment-classification task:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
```

The complete model is then fine-tuned using labeled IMDb reviews.

## 🛠️ Technologies Used

* Python 3.10
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* Seaborn
* tqdm
* CUDA
* NVIDIA GPU

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <https://github.com/rohanbangar2509/Deep-Learning>
```

### 2. Navigate to the project directory

```bash
cd BERT_Sentiment_Analysis_IMDb
```

### 3. Create a virtual environment

```bash
python -m venv .bert_venv
```

### 4. Activate the virtual environment

#### Windows

```bash
.bert_venv\Scripts\activate
```

#### Linux / macOS / WSL

```bash
source .bert_venv/bin/activate
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
BERT_IMDb_Sentiment_Analysis.ipynb
```

Run the notebook sequentially from top to bottom.

## ▶️ Training Using Python Script

The model can also be trained directly using the provided training script:

```bash
python train.py
```

The trained model is saved inside:

```text
outputs/bert-imdb/
```

## 🔍 Sentiment Prediction

After training:

```bash
python predict.py --text "The movie was fantastic, emotional and very well acted."
```

Example output:

```text
Sentiment : POSITIVE
Confidence: 0.98
```

The exact confidence value depends on the trained model.

## 📈 Results

The model is evaluated using:

```text
Accuracy
Precision
Recall
F1-score
```

The notebook also generates a confusion matrix and training/validation performance plots.

### Final Evaluation

After running the experiment, record the actual results:

```text
Test Accuracy : XX.XX%
Precision      : XX.XX%
Recall         : XX.XX%
F1-score       : XX.XX%
```

The final values may vary depending on:

* Training dataset size
* Number of epochs
* Learning rate
* Batch size
* Maximum sequence length
* Random seed
* Hardware
* Model checkpoint

## 📊 Evaluation Metrics

### Accuracy

Accuracy measures the percentage of correctly classified reviews.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Precision measures how many reviews predicted as positive were actually positive.

```text
Precision =
True Positives / (True Positives + False Positives)
```

### Recall

Recall measures how many actual positive reviews were correctly identified.

```text
Recall =
True Positives / (True Positives + False Negatives)
```

### F1-score

The F1-score is the harmonic mean of precision and recall.

```text
F1 =
2 × (Precision × Recall)
------------------------
   Precision + Recall
```

### Confusion Matrix

```text
                  Predicted
               Negative  Positive
Actual
Negative          TN        FP
Positive          FN        TP
```

## 🔄 Project Workflow

```text
IMDb Dataset
      │
      ▼
Load Dataset
      │
      ▼
Explore Dataset
      │
      ▼
Analyze Class Distribution
      │
      ▼
Analyze Review Length
      │
      ▼
Load BERT Tokenizer
      │
      ▼
Tokenize Reviews
      │
      ▼
Create DataLoaders
      │
      ▼
Load Pre-trained BERT
      │
      ▼
Fine-tune BERT
      │
      ▼
Validate Model
      │
      ▼
Evaluate on Test Data
      │
      ▼
Calculate Metrics
      │
      ▼
Generate Confusion Matrix
      │
      ▼
Predict Custom Reviews
      │
      ▼
Save Fine-tuned Model
```

## 📁 Repository Structure

```text
BERT_Sentiment_Analysis_IMDb/
│
├── BERT_IMDb_Sentiment_Analysis.ipynb
│
├── train.py
│
├── predict.py
│
├── outputs/
│   └── bert-imdb/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── ...
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 💾 Trained Model

The fine-tuned BERT model is generated after completing the training process.

```python
model.save_pretrained("outputs/bert-imdb")
tokenizer.save_pretrained("outputs/bert-imdb")
```

The trained model files are not included in the Git repository by default because Transformer model weights can be large.

They can be regenerated by running:

```bash
python train.py
```

## 🧪 Sample Predictions

### Positive Review

```text
"The movie was fantastic, emotional and beautifully acted."
```

Expected sentiment:

```text
POSITIVE
```

### Negative Review

```text
"The movie was boring, predictable and poorly written."
```

Expected sentiment:

```text
NEGATIVE
```

The predictions are generated by the fine-tuned BERT model and may vary depending on the training configuration.

## ⚡ GPU Acceleration

The project automatically uses CUDA when a compatible NVIDIA GPU is available.

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

Example:

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 4050 Laptop GPU
```

If CUDA is unavailable, the project automatically falls back to CPU execution.

A CUDA-enabled NVIDIA GPU is recommended for faster BERT fine-tuning.

## 📚 Key Concepts Demonstrated

* Natural Language Processing
* Transfer Learning
* Transformer Architecture
* BERT
* Pre-trained Language Models
* BERT Tokenization
* Input IDs
* Attention Masks
* Token Type IDs
* Dynamic Padding
* Text Classification
* Sentiment Analysis
* Fine-tuning
* AdamW Optimizer
* Learning Rate Scheduling
* Cross-Entropy Loss
* Batch Processing
* GPU Acceleration
* Model Evaluation
* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* Model Inference

## 🔮 Future Improvements

Possible improvements include:

* Train using the complete IMDb dataset
* Increase the number of training epochs
* Experiment with different learning rates
* Experiment with different batch sizes
* Compare BERT with DistilBERT
* Compare BERT with RoBERTa
* Compare different maximum sequence lengths
* Implement early stopping
* Perform hyperparameter tuning
* Add error analysis for incorrectly classified reviews
* Analyze attention patterns
* Build a Streamlit sentiment-analysis application
* Build a FastAPI inference API
* Deploy the model as a web service
* Compare different Transformer architectures

## 🎯 Learning Outcomes

After completing this project, the learner should understand how a pre-trained Transformer model can be adapted to a downstream NLP task through fine-tuning.

The project demonstrates the practical pipeline:

```text
Pre-trained Language Model
          │
          ▼
Task-specific Dataset
          │
          ▼
Tokenization
          │
          ▼
Fine-tuning
          │
          ▼
Validation
          │
          ▼
Evaluation
          │
          ▼
Inference
```

This provides a foundation for applying Transformer-based models to other NLP tasks such as:

* Text classification
* Spam detection
* Topic classification
* Intent detection
* Emotion classification
* Document classification
* Review classification

## 📚 References

* Hugging Face Transformers Documentation  
  https://huggingface.co/docs/transformers/

* Hugging Face Text Classification Documentation  
  https://huggingface.co/docs/transformers/main/tasks/sequence_classification

* IMDb Dataset — Hugging Face  
  https://huggingface.co/datasets/stanfordnlp/imdb

* BERT — Devlin et al.  
  *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*

## 👨‍💻 Author

**Rohan Bangar**

B.Tech — Artificial Intelligence

---

⭐ If you found this project useful, consider giving the repository a star.

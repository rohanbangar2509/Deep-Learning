# RNN vs LSTM vs GRU for Sequence Classification

A deep learning project that implements and compares **SimpleRNN, LSTM, and GRU** models for binary sequence classification using the **IMDB movie review dataset** with TensorFlow/Keras.

## 📌 Project Overview

This project demonstrates the workflow of building and comparing recurrent neural network architectures for sequence classification:

* Loading the IMDB movie review dataset
* Understanding the dataset and class distribution
* Converting text reviews into integer sequences
* Limiting the vocabulary size
* Padding sequences to a fixed length
* Building an RNN model
* Building an LSTM model
* Building a GRU model
* Training all three models using the same experimental configuration
* Evaluating model performance
* Calculating accuracy, precision, recall, and F1-score
* Generating confusion matrices
* Visualizing training and validation performance
* Comparing model parameters
* Comparing training time
* Analyzing advantages and limitations

The three models use the same embedding dimension, hidden units, dropout, optimizer, loss function, batch size, and number of epochs. The primary difference is the recurrent layer used in each model.

## 🧠 Model Architecture

### RNN

```text
Input Sequence
      │
      ▼
Embedding
      │
      │ 128-dimensional embedding
      ▼
SimpleRNN
      │
      │ 64 units
      ▼
Dropout
      │
      │ 0.5
      ▼
Dense Output Layer
      │
      │ 1 neuron
      │ Sigmoid activation
      ▼
Binary Prediction
      │
      ├── 0 → Negative Review
      └── 1 → Positive Review
```

### LSTM

```text
Input Sequence
      │
      ▼
Embedding
      │
      │ 128-dimensional embedding
      ▼
LSTM
      │
      │ 64 units
      ▼
Dropout
      │
      │ 0.5
      ▼
Dense Output Layer
      │
      │ 1 neuron
      │ Sigmoid activation
      ▼
Binary Prediction
      │
      ├── 0 → Negative Review
      └── 1 → Positive Review
```

### GRU

```text
Input Sequence
      │
      ▼
Embedding
      │
      │ 128-dimensional embedding
      ▼
GRU
      │
      │ 64 units
      ▼
Dropout
      │
      │ 0.5
      ▼
Dense Output Layer
      │
      │ 1 neuron
      │ Sigmoid activation
      ▼
Binary Prediction
      │
      ├── 0 → Negative Review
      └── 1 → Positive Review
```

## 📊 Dataset

The project uses the **IMDB movie review dataset**, which contains movie reviews labeled as either positive or negative.

The dataset contains:

```text
Total Reviews: 50,000
Training Reviews: 25,000
Testing Reviews: 25,000

Classes:
0 → Negative
1 → Positive
```

The dataset is loaded using:

```python
tf.keras.datasets.imdb.load_data(num_words=NUM_WORDS)
```

The vocabulary is limited to the **10,000 most frequent words**.

## ⚙️ Data Preprocessing

The original IMDB reviews are represented as sequences of integer word indices.

The project performs the following preprocessing steps:

1. Load the IMDB dataset.
2. Limit the vocabulary to 10,000 words.
3. Inspect sequence lengths.
4. Pad or truncate sequences to a fixed length of 200.
5. Prepare the data for the recurrent neural network models.

The final input shapes are:

```text
Training Data: (25000, 200)
Testing Data:  (25000, 200)
```

Sequence padding is performed using:

```python
tf.keras.utils.pad_sequences(
    X_train,
    maxlen=MAX_LEN
)

tf.keras.utils.pad_sequences(
    X_test,
    maxlen=MAX_LEN
)
```

## ⚙️ Experimental Configuration

All three models use the same configuration to provide a fair comparison.

```text
Vocabulary Size       : 10,000
Maximum Sequence Len  : 200
Embedding Dimension   : 128
Hidden Units          : 64
Dropout               : 0.5
Optimizer             : Adam
Loss Function         : Binary Crossentropy
Batch Size            : 128
Maximum Epochs        : 5
Validation Split      : 20%
Early Stopping        : Yes
```

Only the recurrent layer is changed:

```text
RNN  → SimpleRNN(64)
LSTM → LSTM(64)
GRU  → GRU(64)
```

## 🛠️ Technologies Used

* Python 3.10
* TensorFlow 2.21.0
* Keras
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
cd rnn-lstm-gru-sequence-classification
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
notebooks/RNN_LSTM_GRU_Sequence_Classification.ipynb
```

Run the notebook from top to bottom.

## 📈 Results

The three models are evaluated using multiple classification metrics:

```text
Accuracy
Precision
Recall
F1-Score
```

The project also compares:

```text
Training Time
Number of Parameters
Validation Accuracy
Validation Loss
Confusion Matrix
```

The final comparison table generated in the notebook contains:

```text
Model
Parameters
Training Time
Accuracy
Precision
Recall
F1-Score
```

The notebook also provides visualizations for:

* Training vs validation accuracy
* Training vs validation loss
* Confusion matrices
* Model parameter comparison
* Training-time comparison

## 🔍 Project Workflow

```text
IMDB Dataset
      │
      ▼
Load Dataset
      │
      ▼
Explore Dataset
      │
      ▼
Convert Reviews to Sequences
      │
      ▼
Limit Vocabulary
      │
      ▼
Pad Sequences
      │
      ▼
Build RNN / LSTM / GRU
      │
      ▼
Compile Models
      │
      ▼
Train Models
      │
      ▼
Generate Predictions
      │
      ▼
Calculate Evaluation Metrics
      │
      ▼
Generate Confusion Matrices
      │
      ▼
Plot Training Curves
      │
      ▼
Compare Parameters & Training Time
      │
      ▼
Analyze Results
```

## 📁 Repository Structure

```text
rnn-lstm-gru-sequence-classification/
│
├── notebooks/
│   └── RNN_LSTM_GRU_Sequence_Classification.ipynb
│
├── models/
│   └── README.md
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## 💾 Trained Models

The trained model files are not included in the Git repository by default.

The models can be saved using the Keras `.keras` format:

```python
rnn_model.save("rnn_model.keras")
lstm_model.save("lstm_model.keras")
gru_model.save("gru_model.keras")
```

The trained model files can be excluded through `.gitignore` to keep the repository lightweight.

## 📚 Key Concepts Demonstrated

This project demonstrates the following deep learning concepts:

* Sequence classification
* Natural Language Processing
* Text preprocessing
* Tokenized sequences
* Sequence padding
* Word embeddings
* Recurrent Neural Networks
* SimpleRNN
* Long Short-Term Memory
* LSTM gates
* Gated Recurrent Units
* GRU gates
* Dropout regularization
* Binary classification
* Sigmoid activation
* Binary cross-entropy
* Adam optimizer
* Early stopping
* Model evaluation
* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* Training and validation curves
* Model parameter comparison
* Training-time comparison

## 🔮 Future Improvements

Possible improvements include:

* Experiment with different vocabulary sizes
* Experiment with different sequence lengths
* Tune the embedding dimension
* Experiment with different numbers of recurrent units
* Compare different dropout rates
* Experiment with different learning rates
* Compare different optimizers
* Increase the number of training epochs
* Perform systematic hyperparameter tuning
* Add Bidirectional RNN/LSTM/GRU models
* Compare with Transformer-based models
* Use pretrained word embeddings
* Deploy the best-performing model as a REST API
* Build a web interface for sentiment prediction

## 👨‍💻 Author

**Rohan Bangar**

B.Tech — Artificial Intelligence

---

⭐ If you found this project useful, consider giving the repository a star.

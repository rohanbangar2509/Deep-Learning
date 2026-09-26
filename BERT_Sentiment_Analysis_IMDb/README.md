# BERT Sentiment Analysis using Hugging Face Transformers

A deep learning project that uses a pre-trained BERT (`bert-base-uncased`) model with PyTorch and Hugging Face Transformers to classify movie reviews as positive or negative using the IMDb dataset.

## 📌 Project Overview

This project demonstrates the complete workflow of applying a pre-trained Transformer model to a Natural Language Processing (NLP) classification problem:

* Loading the IMDb sentiment-analysis dataset
* Exploring the text dataset
* Analyzing class distribution
* Analyzing review lengths
* Loading a pre-trained BERT tokenizer
* Tokenizing text data
* Creating attention masks and input IDs
* Preparing PyTorch DataLoaders
* Loading a pre-trained BERT model
* Adding a classification head
* Fine-tuning BERT on the IMDb dataset
* Evaluating model performance
* Generating classification metrics
* Visualizing the confusion matrix
* Visualizing training and validation performance
* Performing sentiment prediction on custom text
* Saving the fine-tuned BERT model

The project uses transfer learning, where a BERT model that has already learned general language representations from large-scale text is fine-tuned for the specific task of binary sentiment classification.

## 🧠 Model Architecture

The project uses the pre-trained `bert-base-uncased` model followed by a classification layer for two sentiment classes.

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

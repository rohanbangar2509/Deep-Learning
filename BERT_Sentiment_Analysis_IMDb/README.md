# BERT Sentiment Analysis on IMDb

## Assignment
**Implement a pre-trained BERT model for sentiment analysis or text classification on a sample dataset.**

This project fine-tunes `bert-base-uncased` for binary sentiment classification using the IMDb movie-review dataset.

## Dataset
The project uses the Hugging Face dataset:

`stanfordnlp/imdb`

Labels:
- `0` = Negative
- `1` = Positive

## Model
- Pre-trained model: `bert-base-uncased`
- Task: Binary text classification
- Fine-tuning framework: PyTorch + Hugging Face Transformers

## Project Structure

```text
BERT_Sentiment_Analysis_IMDb/
├── BERT_IMDb_Sentiment_Analysis.ipynb
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── outputs/
    └── bert-imdb/
```

## Installation

```bash
python -m venv .bert_venv
```

Windows:
```bash
.bert_venv\Scripts\activate
```

Linux / WSL:
```bash
source .bert_venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

The default experiment uses:
- 8,000 training reviews
- 2,000 validation reviews
- 2,000 test reviews
- Maximum sequence length: 256
- Batch size: 8
- Epochs: 2
- Learning rate: 2e-5

These settings are intentionally suitable for student hardware. For a larger experiment, increase the sample sizes and/or epochs in `train.py`.

## Inference

After training:

```bash
python predict.py --text "The movie was fantastic, emotional and very well acted."
```

## Notebook
`BERT_IMDb_Sentiment_Analysis.ipynb` contains the complete step-by-step implementation:
1. Objective and theory
2. Environment setup
3. Dataset loading
4. Dataset exploration
5. BERT tokenizer
6. Tokenization and padding
7. Model initialization
8. Fine-tuning
9. Validation
10. Test evaluation
11. Confusion matrix
12. Training curves
13. Custom review prediction
14. Conclusion

## Expected Academic Conclusion
The experiment demonstrates how a pre-trained BERT language model can be fine-tuned for downstream sentiment classification. BERT's contextual representation allows the model to capture semantic information from movie reviews and classify them as positive or negative. The final accuracy, precision, recall, F1-score and confusion matrix should be reported from the actual run rather than hard-coded.

## References
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers/main/tasks/sequence_classification
- IMDb dataset: https://huggingface.co/datasets/stanfordnlp/imdb

# LSTM-Based Sales Forecasting using TensorFlow/Keras

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#lstm-based-sales-forecasting-using-tensorflowkeras)

A deep learning project that uses a Long Short-Term Memory (LSTM) neural network built with TensorFlow/Keras to forecast daily sales using historical time-series data.

## 📌 Project Overview

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#-project-overview)

This project demonstrates the complete workflow of building an LSTM-based model for time-series forecasting:

* Loading historical sales data
* Exploring the dataset
* Converting dates into time-series format
* Visualizing sales trends
* Splitting data chronologically into training and testing sets
* Normalizing sales values
* Creating time-series sequences using a sliding window
* Building an LSTM neural network using Keras
* Compiling the model
* Training the model
* Generating sales predictions
* Evaluating model performance using MAE and RMSE
* Visualizing actual vs predicted sales
* Saving the trained model and prediction results

The model uses the previous **7 days of sales** to predict the sales for the following day.

## 🧠 Model Architecture

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#-model-architecture)

```text
Historical Sales
       │
       │ Previous 7 Days
       ▼
   LSTM Layer
       │
       │ 50 units
       ▼
   LSTM Layer
       │
       │ 50 units
       ▼
  Dense Output
       │
       │ 1 unit
       ▼
Next-Day Sales Prediction
```

**svg**

The model consists of two LSTM layers followed by a Dense output layer.

### Model Configuration

```text
Lookback Window : 7 days
LSTM Layers     : 2
LSTM Units      : 50
Output Units    : 1
Optimizer       : Adam
Loss Function   : Mean Squared Error
Epochs          : 50
Batch Size      : 32
```

## 📊 Dataset

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#-dataset)

The project uses the **Store Item Demand Forecasting** dataset.

For this project, the sales history of **Store 1 and Item 1** was selected to create a focused univariate time-series forecasting problem.

The resulting dataset contains:

```text
1,826 daily sales observations
2013-01-01 to 2017-12-31
```

The working dataset contains two columns:

```text
date
sales
```

**svg**

The dataset is stored in:

```text
data/sales.csv
```

## ⚙️ Data Preprocessing

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#%EF%B8%8F-data-preprocessing)

The following preprocessing steps were performed:

### 1. Date Conversion

The `date` column was converted into a datetime format.

```python
df["date"] = pd.to_datetime(df["date"])
```

### 2. Chronological Sorting

The observations were sorted according to their date to preserve the time-series order.

### 3. Train-Test Split

The dataset was divided chronologically:

```text
80% → Training Data
20% → Testing Data
```

Random shuffling was not used because the temporal order of observations is important in time-series forecasting.

### 4. Normalization

Sales values were normalized to the range `0` to `1` using `MinMaxScaler`.

```python
scaler = MinMaxScaler(feature_range=(0, 1))
```

### 5. Sequence Generation

A lookback window of 7 days was used.

```text
Day 1 ─┐
Day 2  │
Day 3  │
Day 4  ├──→ LSTM → Day 8 Prediction
Day 5  │
Day 6  │
Day 7 ─┘
```

This converts the time-series data into sequences suitable for LSTM training.

## 🛠️ Technologies Used

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#%EF%B8%8F-technologies-used)

* Python 3.10+
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## 🚀 Getting Started

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#-getting-started)

### 1. Clone the repository

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#1-clone-the-repository)

```bash
git clone https://github.com/rohanbangar2509/Deep-Learning.git
```

**svg**

### 2. Navigate to the project directory

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#2-navigate-to-the-project-directory)

```bash
cd Deep-Learning/lstm-sales-forecasting
```

**svg**

### 3. Create a virtual environment

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#3-create-a-virtual-environment)

```bash
python -m venv venv
```

**svg**

### 4. Activate the virtual environment

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#4-activate-the-virtual-environment)

#### Windows

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#windows)

```bash
venv\Scripts\activate
```

**svg**

#### Linux/macOS

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#linuxmacos)

```bash
source venv/bin/activate
```

**svg**

### 5. Install dependencies

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#5-install-dependencies)

```bash
pip install -r requirements.txt
```

**svg**

### 6. Launch Jupyter Notebook

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#6-launch-jupyter-notebook)

```bash
jupyter notebook
```

**svg**

Open:

```text
notebooks/LSTM_Sales_Forecasting.ipynb
```

**svg**

Run the notebook from top to bottom.

### 7. Run the training script

The model can also be trained directly using the Python script:

```bash
python src/train.py
```

**svg**

The trained model and prediction results are saved in the `results/` directory.

## 📈 Results

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#-results)

The trained LSTM model achieved the following results on the test dataset:

```text
MAE  : 4.3993
RMSE : 5.5430
```

**svg**

### Mean Absolute Error — MAE

MAE represents the average absolute difference between the actual and predicted sales values.

```text
MAE = 4.3993
```

### Root Mean Squared Error — RMSE

RMSE measures the prediction error while giving greater importance to larger errors.

```text
RMSE = 5.5430
```

**svg**

### Training and Validation Loss

![Training Loss](results/training_loss.png)

### Actual vs Predicted Sales

![Actual vs Predicted Sales](results/actual_vs_predicted.png)

## 🔍 Project Workflow

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#-project-workflow)

```text
Sales Dataset
      │
      ▼
Load Dataset
      │
      ▼
Convert Date
      │
      ▼
Sort Chronologically
      │
      ▼
Train-Test Split
      │
      ▼
Normalize Sales
      │
      ▼
Create 7-Day Sequences
      │
      ▼
Build LSTM Model
      │
      ▼
Compile Model
      │
      ▼
Train Model
      │
      ▼
Generate Predictions
      │
      ▼
Calculate MAE & RMSE
      │
      ▼
Visualize Results
      │
      ▼
Save Model & Results
```

**svg**

## 📁 Repository Structure

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#-repository-structure)

```text
lstm-sales-forecasting/
│
├── data/
│   └── sales.csv
│
├── notebooks/
│   └── LSTM_Sales_Forecasting.ipynb
│
├── results/
│   ├── actual_vs_predicted.png
│   ├── training_loss.png
│   ├── metrics.csv
│   ├── predictions.csv
│   └── lstm_sales_model.keras
│
├── src/
│   └── train.py
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

**svg**

## 💾 Trained Model

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#-trained-model)

The trained LSTM model is saved in the `.keras` format:

```text
results/lstm_sales_model.keras
```

**svg**

The model can be loaded using:

```python
from tensorflow.keras.models import load_model

model = load_model("results/lstm_sales_model.keras")
```

The repository also contains:

```text
results/metrics.csv
results/predictions.csv
```

which store the evaluation metrics and actual versus predicted values.

## 📚 Key Concepts Demonstrated

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#-key-concepts-demonstrated)

This project demonstrates the following deep learning and time-series concepts:

* Time-Series Forecasting
* Historical Data Analysis
* Data Preprocessing
* Chronological Train-Test Split
* Feature Scaling
* Min-Max Normalization
* Sliding Window Sequences
* Lookback Window
* LSTM Neural Networks
* Sequential Model Architecture
* Dense Output Layer
* Adam Optimizer
* Mean Squared Error
* Model Training
* Model Validation
* Sales Prediction
* Mean Absolute Error
* Root Mean Squared Error
* Prediction Visualization
* Model Saving
* Git and GitHub

## 🔮 Future Improvements

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#-future-improvements)

Possible improvements include:

* Predict multiple future days instead of only the next day
* Experiment with different lookback windows
* Compare LSTM with GRU architectures
* Add Dropout for regularization
* Perform hyperparameter tuning
* Compare different optimizers
* Include additional features such as holidays and promotions
* Perform multivariate time-series forecasting
* Compare LSTM predictions with traditional forecasting methods
* Build a simple web interface for sales forecasting
* Deploy the forecasting model as an API

## 👨‍💻 Author

[svg](https://github.com/rohanbangar2509/Deep-Learning/tree/main/lstm-sales-forecasting#%E2%80%8D-author)

**Rohan Bangar**

B.Tech — Artificial Intelligence

---

⭐ If you found this project useful, consider giving the repository a star.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Configuration
LOOKBACK = 7
STORE_ID = 1
ITEM_ID = 1
EPOCHS = 50
BATCH_SIZE = 32


# Load and prepare the data

# Load dataset
df = pd.read_csv("data/train.csv")

# Convert date
df["date"] = pd.to_datetime(df["date"])

# Select one store and one item
sales_df = df[
    (df["store"] == STORE_ID) &
    (df["item"] == ITEM_ID)
].copy()

# Sort chronologically
sales_df = sales_df.sort_values("date")

# Set date as index
sales_df.set_index("date", inplace=True)

# Keep only sales
sales = sales_df[["sales"]]


# Train/Test split

# Train-test split
train_size = int(len(sales) * 0.8)

train = sales.iloc[:train_size]
test = sales.iloc[train_size:]


# Normalize

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))

train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)


# Sequence function

def create_sequences(data, lookback):
    X = []
    y = []

    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])

    return np.array(X), np.array(y)


X_train, y_train = create_sequences(
    train_scaled,
    LOOKBACK
)

test_input = np.concatenate(
    (train_scaled[-LOOKBACK:], test_scaled)
)

X_test, y_test = create_sequences(
    test_input,
    LOOKBACK
)

# Build the LSTM

# Build LSTM model
model = Sequential([
    LSTM(
        50,
        return_sequences=True,
        input_shape=(LOOKBACK, 1)
    ),
    LSTM(50),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# Train

# Train model
history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

# Predict

# Make predictions
predictions_scaled = model.predict(X_test)

# Convert back to original scale
predictions = scaler.inverse_transform(
    predictions_scaled
)

actual = scaler.inverse_transform(
    y_test
)

# Evaluate

# Evaluation
mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Save the model

# Save trained model
model.save("results/lstm_sales_model.keras")

# Save metrics

metrics = pd.DataFrame({
    "Metric": ["MAE", "RMSE"],
    "Value": [mae, rmse]
})

metrics.to_csv(
    "results/metrics.csv",
    index=False
)

# Save predictions

results = pd.DataFrame({
    "Actual": actual.flatten(),
    "Predicted": predictions.flatten()
})

results.to_csv(
    "results/predictions.csv",
    index=False
)

# Save graphs

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error")
plt.legend()

plt.savefig(
    "results/training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Actual vs predicted:

plt.figure(figsize=(14, 6))

plt.plot(
    actual,
    label="Actual Sales"
)

plt.plot(
    predictions,
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Days")
plt.ylabel("Sales")
plt.legend()

plt.savefig(
    "results/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



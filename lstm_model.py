# lstm_model.py
# Predicting Federal Funds Effective Rate for 2025 using LSTM
# Clear, simple, and minimalist implementation inspired by great minds

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 1. Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv("finaldata.csv")  # Dataset must be in the same directory
TARGET = 'FEDFUNDS'  # The column we're trying to predict

# Features and target
features = data.drop(columns=[TARGET]).values  # Drop target column
target = data[TARGET].values  # Extract target column

# Scale features for LSTM
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features)

# 2. Prepare sequences for LSTM
print("Preparing data sequences for LSTM...")
SEQ_LEN = 12  # Sequence length: past 12 months
X, y = [], []

for i in range(SEQ_LEN, len(scaled_features)):
    X.append(scaled_features[i-SEQ_LEN:i])  # Append last 12 months
    y.append(target[i])  # Append the target value for this sequence

X, y = np.array(X), np.array(y)  # Convert to NumPy arrays

# Split into training (80%) and testing (20%) sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

# 3. Build the LSTM model
print("Building LSTM model...")
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),  # Regularization
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)  # Output: a single value (interest rate)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 4. Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 5. Evaluate the model
print("Evaluating the model...")
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error on test set: {mse:.4f}")

# 6. Visualize the predictions
print("Visualizing predictions...")
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Rates')
plt.plot(predictions, label='Predicted Rates')
plt.title('Federal Funds Rate Prediction')
plt.xlabel('Time')
plt.ylabel('Interest Rate')
plt.legend()
plt.show()

# 7. Save the trained model
print("Saving the trained model...")
model.save("lstm_model.h5")
print("Model saved as 'lstm_model.h5'")

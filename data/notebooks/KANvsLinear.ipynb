!pip install pykan scikit-learn matplotlib numpy pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kan import KAN
import torch
data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print(X.head())
print("Shape:", X.shape)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to numpy arrays for plotting and scaling
X_train_np = X_train.values
X_test_np = X_test.values

# Before scaling
plt.scatter(X_train_np[:, 0], X_train_np[:, 1])
plt.title("Before Scaling")
plt.show()

# Scaling (KAN benefits from normalized inputs)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_np)
X_test = scaler.transform(X_test_np)

# After scaling
plt.scatter(X_train[:, 0], X_train[:, 1])
plt.title("After Scaling")
plt.show()
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

y_pred_linear = linear_model.predict(X_test)

rmse_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
r2_linear = r2_score(y_test, y_pred_linear)

print("Linear Model Results:")
print("RMSE:", rmse_linear)
print("R2 Score:", r2_linear)

# Convert to torch
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)

X_test_t = torch.tensor(X_test, dtype=torch.float32)

# Define KAN
model = KAN(width=[X_train.shape[1], 20, 1], grid=5, k=3)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Manual training loop
num_steps = 1000
for i in range(num_steps):
    # Forward pass
    output = model(X_train_t)
    # Compute loss
    loss = torch.mean((output - y_train_t)**2) # MSE Loss

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
        print(f'Step [{i+1}/{num_steps}], Loss: {loss.item():.4f}')

# Predict
y_pred_kan = model(X_test_t).detach().numpy().ravel()

rmse_kan = np.sqrt(mean_squared_error(y_test, y_pred_kan))
r2_kan = r2_score(y_test, y_pred_kan)

print("\nKAN Model Results:")
print("RMSE:", rmse_kan)
print("R2 Score:", r2_kan)

plt.figure()

plt.scatter(y_test, y_pred_linear, alpha=0.5, label="Linear")
plt.scatter(y_test, y_pred_kan, alpha=0.5, label="KAN")

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("KAN vs Linear Regression (Real Dataset)")
plt.legend()

plt.show()

plt.figure()

linear_errors = y_test - y_pred_linear
kan_errors = y_test - y_pred_kan

plt.hist(linear_errors, bins=30, alpha=0.5, label="Linear")
plt.hist(kan_errors, bins=30, alpha=0.5, label="KAN")

plt.title("Error Distribution")
plt.legend()
plt.show()
print("\n=== Final Comparison ===")
print(f"{'Model':<20}{'RMSE':<10}{'R2 Score'}")
print(f"{'Linear Regression':<20}{rmse_linear:<10.4f}{r2_linear:.4f}")
print(f"{'KAN Network':<20}{rmse_kan:<10.4f}{r2_kan:.4f}")

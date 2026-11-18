import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model using gradient descent."""
        n_samples, n_features = X.shape

        # 1. Initialize weights and bias to zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # 2. Gradient Descent optimization loop
        for _ in range(self.n_iters):
            # Compute current model predictions
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute partial derivatives (gradients)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update weights and bias parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for given input features."""
        return np.dot(X, self.weights) + self.bias


# --- Fix 1: Read your CSV file ---
df = pd.read_csv("salary_dataset.csv")

# --- Fix 2: Extract columns as 2D NumPy arrays instead of 1D lists ---
# Using df[['YOE']] ensures a 2D matrix structure [n_samples, 1_feature]
X = df[['YOE']].values
y = df['Salary'].values

# --- Fix 3: Set an appropriate learning rate ---
# Because Salary values are massive compared to YOE, a high lr like 0.01
# will blow up the numbers into 'NaN'. We use a smaller lr or scale.
model = LinearRegression(lr=0.001, n_iters=5000)

# Train the model with array variables
model.fit(X, y)

# --- Fix 4: Format input matrix for predictions ---
# Pass 4 inside a 2D array representation: [[4]]
query_point = np.array([[10]])
prediction = model.predict(query_point)

print(f"Predicted Salary for  YOE: {prediction[0]:.2f}")

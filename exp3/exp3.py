import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load and preprocess dataset

df = pd.read_csv('titanic.csv')

print(df.head())
print(df.info())

df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], dtype=int)

# 2. Separate features and target

X = df.drop('Survived', axis=1).values.astype(float)
y = df['Survived'].values.astype(float)

# 3. Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Standardize features

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# 5. Sigmoid function

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# 6. Cost function

def compute_cost(X, y, weights, bias):
    m = X.shape[0]

    z = np.dot(X, weights) + bias
    predictions = sigmoid(z)

    epsilon = 1e-10
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    cost = (-1 / m) * np.sum(
        y * np.log(predictions) +
        (1 - y) * np.log(1 - predictions)
    )

    return cost


# 7. Gradient Descent

def gradient_descent(X, y, learning_rate, iterations):
    m, n = X.shape

    weights = np.zeros(n)
    bias = 0.0

    costs = []

    for i in range(iterations):

        z = np.dot(X, weights) + bias
        predictions = sigmoid(z)

        dz = predictions - y

        dw = (1 / m) * np.dot(X.T, dz)
        db = (1 / m) * np.sum(dz)

        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        costs.append(compute_cost(X, y, weights, bias))

    return weights, bias, costs


# 8. Prediction function

def predict(X, weights, bias):
    probabilities = sigmoid(np.dot(X, weights) + bias)

    return (probabilities >= 0.5).astype(int)


# 9. Evaluation metrics

def evaluate_metrics(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / len(y_true)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return accuracy, precision, recall, f1, TP, TN, FP, FN


# 10. Train the model

learning_rate = 0.01
iterations = 20000

weights, bias, costs = gradient_descent(
    X_train, y_train, learning_rate, iterations
)


# 11. Make predictions

y_pred = predict(X_test, weights, bias)


# 12. Evaluate model

accuracy, precision, recall, f1, TP, TN, FP, FN = evaluate_metrics(
    y_test, y_pred
)

print("\nLogistic Regression Results")

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

print("\nConfusion Matrix Components")

print("True Positives:", TP)
print("True Negatives:", TN)
print("False Positives:", FP)
print("False Negatives:", FN)

# ==========================================
# K-NEAREST NEIGHBOURS (KNN) FROM SCRATCH
# ==========================================

import random
import math
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and Prepare Data
data = pd.read_csv("iris.csv")
print("First few rows of the dataset:")
print(data.head())

# Extract features and labels
X = data[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values.tolist()
y = data["Species"].tolist()
dataset = list(zip(X, y))

# Shuffle and split into Train (70%) and Test (30%) sets
random.seed(42)
random.shuffle(dataset)
n = int(0.7 * len(dataset))
train_data = dataset[:n]
test_data = dataset[n:]


# 2. Define KNN Core Functions
def distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def get_neighbours(point, k):
    """Finds the k nearest neighbours from the training set for a given point."""
    distances = [
        (distance(x, point), label)
        for x, label in train_data
    ]
    return sorted(distances)[:k]



def predict(point, k):
    """Predicts the class label based on majority vote of the k nearest neighbours."""
    neighbours = get_neighbours(point, k)
    labels = [label for distance_value, label in neighbours]
    return max(set(labels), key=labels.count)


# 3. Evaluate Model on Test Data
k = 5
actual = [label for x, label in test_data]
predicted = [predict(x, k) for x, label in test_data]

correct = sum(a == p for a, p in zip(actual, predicted))
print("\n--- Evaluation Results ---")
print("Correct:", correct)
print("Incorrect:", len(actual) - correct)
print(f"Accuracy: {correct / len(actual) * 100:.2f}%")


# 4. Generate Confusion Matrix
classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
cm = [[0] * 3 for _ in range(3)]

for a, p in zip(actual, predicted):
    cm[classes.index(a)][classes.index(p)] += 1

print("\nConfusion Matrix:")
for row in cm:
    print(row)


# 5. Plot 1: Full Dataset Scatter Plot
plt.figure(figsize=(8, 6))
for name, marker in zip(classes, ["o", "s", "^"]):
    species_data = data[data["Species"] == name]
    plt.scatter(
        species_data["PetalLengthCm"],
        species_data["PetalWidthCm"],
        marker=marker,
        label=name
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Dataset - Petal Dimensions")
plt.legend()
plt.grid(True)
plt.show()


# 6. Test a Single Point & Plot KNN Neighbourhood
test_point, actual_class = test_data[0]
predicted_class = predict(test_point, k)
neighbours = get_neighbours(test_point, k)

print("\n--- Single Test Point Prediction ---")
print("Test Point (Features):", test_point)
print("Actual Class:", actual_class)
print("Predicted Class:", predicted_class)
print("Nearest Neighbours:", neighbours)

# Plotting the neighbourhood for the specific test point
plt.figure(figsize=(8, 6))

# Plot all training points
for x, label in train_data:
    plt.scatter(
        x[2], x[3],
        marker=["o", "s", "^"][classes.index(label)],
        color="gray",
        alpha=0.6
    )

# Highlight the k-nearest neighbours with a black ring
for distance_value, features, label in sorted(
    [
        (distance(x, test_point), x, label)
        for x, label in train_data
    ]
)[:k]:
    plt.scatter(
        features[2], features[3],
        s=200,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5
    )

# Plot the test point itself as a star
plt.scatter(
    test_point[2], test_point[3],
    s=300,
    marker="*",
    color="red",
    label="Test Point"
)

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title(f"KNN Visualization (k={k})")
plt.legend()
plt.grid(True)
plt.show()

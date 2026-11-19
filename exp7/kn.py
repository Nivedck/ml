import random
import math
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("iris.csv")
print("First few rows of the dataset:")
print(data.head())

X = data[["sepal_length","sepal_width","petal_length","petal_width"]].values.tolist()
y = data["species"].tolist()
dataset = list(zip(X, y))


random.seed(42)
random.shuffle(dataset)
split_index = int(0.7 * len(dataset))
train_data = dataset[:split_index]
test_data = dataset[split_index:]



def distance(p1, p2):
    total = 0
    for i in range(len(p1)):
        diff = p1[i] - p2[i]
        total += diff ** 2
    return math.sqrt(total)


def get_neighbours(point, k):
    distances = []
    for x, label in train_data:
        d = distance(x, point)
        distances.append((d, label))
    distances.sort()
    return distances[:k]


def predict(point, k):
    neighbours = get_neighbours(point, k)
    counts = {}

    for dist, label in neighbours:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1

    best_label = None
    highest_count = -1
    for label, count in counts.items():
        if count > highest_count:
            highest_count = count
            best_label = label

    return best_label


# 3. Evaluate Model on Test Data
k = 5
correct = 0
actual = []
predicted = []

for x, label in test_data:
    pred = predict(x, k)
    actual.append(label)
    predicted.append(pred)
    if pred == label:
        correct += 1

total_test = len(test_data)
print("\n--- Evaluation Results ---")
print("Correct:", correct)
print("Incorrect:", total_test - correct)
print(f"Accuracy: {(correct / total_test) * 100:.2f}%")



classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
cm = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]

for i in range(len(actual)):
    row = classes.index(actual[i])
    col = classes.index(predicted[i])
    cm[row][col] += 1

print("\nConfusion Matrix:")
for row in cm:
    print(row)



markers = ["o", "s", "^"]
plt.figure(figsize=(8, 6))

for i in range(len(classes)):
    name = classes[i]
    species_data = data[data["Species"] == name]
    plt.scatter(
        species_data["PetalLengthCm"],
        species_data["PetalWidthCm"],
        marker=markers[i],
        label=name
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Iris Dataset - Petal Dimensions")
plt.legend()
plt.grid(True)
plt.show()


test_point, actual_class = test_data[0]
predicted_class = predict(test_point, k)
neighbours = get_neighbours(test_point, k)

print("\n--- Single Test Point Prediction ---")
print("Test Point (Features):", test_point)
print("Actual Class:", actual_class)
print("Predicted Class:", predicted_class)
print("Nearest Neighbours:", neighbours)

plt.figure(figsize=(8, 6))


for x, label in train_data:
    idx = classes.index(label)
    plt.scatter(x[2], x[3], marker=markers[idx], color="gray", alpha=0.6)

all_neighbours = []
for x, label in train_data:
    d = distance(x, test_point)
    all_neighbours.append((d, x, label))
all_neighbours.sort()
nearest_k = all_neighbours[:k]

for d, features, label in nearest_k:
    plt.scatter(
        features[2], features[3],
        s=200,
        facecolors="none",
        edgecolors="black",
        linewidths=1.5
    )


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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, log_loss,
    confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)

# 1. Load dataset
df = pd.read_csv("fashion-mnist.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 3. Class names
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 4. Display sample images
plt.figure(figsize=(10, 5))

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i].reshape(28, 28), cmap="gray")
    plt.title(class_names[y_train[i]])
    plt.axis("off")

plt.tight_layout()
plt.show()

# 5. Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# 6. Create and train MLP model
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=50,
    random_state=42
)

model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate model
print("\nTest Loss:", log_loss(y_test, model.predict_proba(X_test)))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 9. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
).plot(cmap="Blues", xticks_rotation=45)

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# 10. Classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=class_names
))

# 11. Display one prediction
i = 0

print("\nActual:", class_names[y_test[i]])
print("Predicted:", class_names[y_pred[i]])

plt.imshow(X_test[i].reshape(28, 28), cmap="gray")
plt.title("Predicted: " + class_names[y_pred[i]])
plt.axis("off")
plt.show()

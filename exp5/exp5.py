
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Function to train and evaluate the model
def evaluate_tree(criterion, name):

    model = DecisionTreeClassifier(
        criterion=criterion, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Confusion Matrix - " + name)
    plt.show()

    # Classification Report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Visualize Decision Tree
    plt.figure(figsize=(12, 8))
    plot_tree(
        model,
        feature_names=iris.feature_names,
        class_names=iris.target_names,
        filled=True
    )
    plt.title(name)
    plt.show()

    return accuracy_score(y_test, y_pred)


# 4. ID3 using Entropy
id3_accuracy = evaluate_tree('entropy', 'ID3 (Entropy)')

# 5. CART using Gini
cart_accuracy = evaluate_tree('gini', 'CART (Gini)')

# 6. Compare accuracy
print("\nComparison of ID3 and CART")
print("ID3 Accuracy:", id3_accuracy)
print("CART Accuracy:", cart_accuracy)

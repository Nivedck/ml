
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
from sklearn.decomposition import TruncatedSVD


# Load Dataset
data = pd.read_csv("spam_ham_dataset.csv")

print("=" * 60)
print("SPAM EMAIL DATASET")
print("=" * 60)

print(data.head())
print("\nDataset Shape:", data.shape)
print("\nColumn Names:", data.columns)
print("\nClass Distribution:\n", data["label"].value_counts())
print("\nMissing Values:\n", data.isnull().sum())

data = data.dropna()

X, y = data["text"], data["label"]


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Emails:", len(X_train))
print("Testing Emails:", len(X_test))


# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining TF-IDF Shape:", X_train_tfidf.shape)
print("Testing TF-IDF Shape:", X_test_tfidf.shape)


# Train SVM Model
svm_model = SVC(kernel="linear", C=1)
svm_model.fit(X_train_tfidf, y_train)

y_pred = svm_model.predict(X_test_tfidf)


# Model Performance
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("Accuracy:", accuracy)
print("Accuracy Percentage:", round(accuracy * 100, 2), "%")


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

print("\nConfusion Matrix:\n", cm)

ConfusionMatrixDisplay(
    cm,
    display_labels=["Ham", "Spam"]
).plot()

plt.title("Confusion Matrix - SVM Spam Mail Classifier")
plt.show()


# Classification Report
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred))


# Classify New Emails
new_emails = [
    "Congratulations! You won $5000. Click here to claim your reward.",
    "Dear Students, Machine Learning exam is tomorrow at 10 AM.",
    "WINNER! You have won a FREE vacation. Claim your prize now.",
    "Please find attached the project report. Let me know if corrections are needed.",
    "Limited time offer! Get a free mobile phone today.",
    "Dear Faculty, department meeting will be held at 2 PM."
]

new_predictions = svm_model.predict(
    vectorizer.transform(new_emails)
)

print("\nNEW EMAIL CLASSIFICATION")

for email, prediction in zip(new_emails, new_predictions):
    print("\nEmail:", email)
    print(
        "Prediction:",
        "SPAM MAIL" if prediction == "spam" else "HAM / NOT SPAM"
    )


# Reduce Dimensions using SVD
svd = TruncatedSVD(n_components=2, random_state=42)
X_2d = svd.fit_transform(X_train_tfidf)


# Train SVM for Visualization
svm_visual = SVC(kernel="linear", C=1)
svm_visual.fit(X_2d, y_train)

y_numeric = np.where(y_train == "spam", 1, 0)


# Create Mesh Grid
x_min = X_2d[:, 0].min() - 0.5
x_max = X_2d[:, 0].max() + 0.5

y_min = X_2d[:, 1].min() - 0.5
y_max = X_2d[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)


# Predict Decision Boundary
Z = svm_visual.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = np.where(Z == "spam", 1, 0).reshape(xx.shape)


# Plot Classification Results
plt.figure(figsize=(10, 7))

plt.contourf(xx, yy, Z, alpha=0.25)

plt.scatter(
    X_2d[y_numeric == 0, 0],
    X_2d[y_numeric == 0, 1],
    label="Ham",
    marker="o",
    s=30
)

plt.scatter(
    X_2d[y_numeric == 1, 0],
    X_2d[y_numeric == 1, 1],
    label="Spam",
    marker="^",
    s=40
)


# Highlight Support Vectors
plt.scatter(
    svm_visual.support_vectors_[:, 0],
    svm_visual.support_vectors_[:, 1],
    s=100,
    facecolors="none",
    edgecolors="black",
    label="Support Vectors"
)

plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.title("SVM Classification of Spam and Ham Emails")

plt.legend()
plt.grid(True)
plt.show()


print("\nPROGRAM EXECUTION COMPLETED")


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# 1. Load dataset
data = pd.read_csv("spam_ham_dataset.csv")
data = data.dropna()

X = data["text"]
y = data["label"]

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# 4. Train SVM model
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# 5. Prediction
y_pred = model.predict(X_test)

# 6. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
print("Confusion Matrix:\n", cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham", "Spam"]
).plot()

plt.show()

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Predict new emails
new_emails = [
    "Congratulations! You won a free prize. Claim now!",
    "Dear students, the exam will be held tomorrow.",
]

new_emails_tfidf = vectorizer.transform(new_emails)
predictions = model.predict(new_emails_tfidf)

for email, prediction in zip(new_emails, predictions):
    print("\nEmail:", email)
    print("Prediction:", prediction.upper())


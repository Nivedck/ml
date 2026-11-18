import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,accuracy_score


df = pd.read_csv("spam_ham_dataset.csv")

X = df['text']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

vectorizer = TfidfVectorizer(
    stop_words="english"
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = LinearSVC()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

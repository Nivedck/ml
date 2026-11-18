import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv("spam.csv",encoding="latin-1")

df = df.iloc[:,:2]
df.columns=["label","text"]

X=df['text']
y=df['label']

print(X.head())
print(y.head())

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

vectorizer = TfidfVectorizer( stop_words = "english")

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model = LinearSVC()

model.fit(X_train,y_train)

new_email = vectorizer.transform([""])
pred = model.predict(new_email)

print(pred)


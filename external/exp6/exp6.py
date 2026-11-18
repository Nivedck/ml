import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score , confusion_matrix , ConfusionMatrixDisplay

df = pd.read_csv("spam_ham_dataset.csv")

print("Email Dataset")
print(df.head())
print(df.info())

df = df.dropna()

X = df['text']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

vectorizer = TfidfVectorizer(
    lower_case=True,
    stop_words="english"
)

X_train = vectorizer.fit_transform(X_train)
X_test  = vectorizer.transform(X_test)

model = SVC()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)


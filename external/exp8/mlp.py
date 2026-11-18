import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report , ConfusionMatrixDisplay


df = pd.read_csv("mnist_train.csv")

print(df.head())

X = df.iloc[:,1:].values
y = df.iloc[:,0].values


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)

plt.imshow(X_train[0].reshape(28, 28), cmap="gray")

plt.tight_layout()
plt.show()

model = MLPClassifier(
    hidden_layer_sizes=(128,64),
    max_iter=50
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

ConfusionMatrixDisplay(cm).plot()
plt.show()

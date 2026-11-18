import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report,accuracy_score

df = pd.read_csv("iris.csv")

print(df.info())
print(df.head())

X = df.drop('species',axis=1)
y = df['species']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

def evaluate_tree(name,criterion):

    model = DecisionTreeClassifier(
        criterion=criterion,
        random_state=42
    )

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    report = classification_report(y_pred , y_test)

    plot_tree(
        model,
        filled=True
    )
    plt.plot()
    plt.show()

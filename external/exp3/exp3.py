import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split

df = pd.read_csv("titanic.csv")

print(df.head())
print(df.info())

df = df.drop(
    ['Name', 'Ticket', 'Cabin', 'PassengerId'],
    axis=1
)

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = pd.get_dummies(df,columns=['Sex','Embarked'],dtype=int)


X = df.drop('Survived',axis=1).values.astype(float)
y = df['Survived'].values.astype(float)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std==0]=1

X_train = (X_train-mean)/std
X_test = (X_test-mean)/std
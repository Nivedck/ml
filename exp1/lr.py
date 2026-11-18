from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("temp-lemon-cleaned.csv")

print(df.head())
print()

X = df[['Temp']].values
y = df[['Lemonde']].values

model = LinearRegression()

model.fit(X,y)

pred = model.predict([[20]])

print(pred)

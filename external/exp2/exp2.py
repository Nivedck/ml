import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


df = pd.read_csv("dataset.csv")

print(df.head())
print(df.isnull().sum())

df = df.dropna()
df = df.drop_duplicates()

X = df[['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']]
y = df['median_house_value']

X_train,X_test , y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print("Model Evaluaton")
print("MSE:",mse)
print("R2 Score:",r2)

print("Enter Values for testing")
longitude = float(input("Enter longitude"))
latitude = float(input("Enter latitude"))
age = float(input("Enter housing median age:"))
rooms = float(input("Enter total rooms:"))
bedrooms = float(input("Enter total bedrooms:"))
population =float(input("Enter population"))
households = float(input("Enter households"))
median_income = float(input("Enter median income"))

new_data = pd.DataFrame({
    "longitude":[longitude],
    "latitude":[latitude],
    "housing_median_age":[age],
    "total_rooms":[rooms],
    "total_bedrooms":[bedrooms],
    "population":[population],
    "households":[households],
    "median_income":[median_income]
})

result = model.predict(new_data)

print("Predicted Value:",result[0])

plt.figure(figsize=(6,6))
plt.scatter(y_test,y_pred,color="blue")
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color="red")
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('dataset.csv')

print("Original Dataset:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

df = df.drop_duplicates()
df = df.dropna()

print("\nDataset after cleaning:")
print(df.head())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

X = df[['longitude',
        'latitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income']]

y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2 Score): {r2:.2f}")

longitude = float(input("\nEnter Longitude: "))
latitude = float(input("Enter Latitude: "))
housing_median_age = float( input("Enter Housing median Age: "))
total_rooms = float(input("Enter Total Rooms: "))
total_bedrooms = float(   input("Enter Total Bedrooms: "))
population = float(input("Enter Population: "))
households = float(input("Enter Households: "))
median_income = float(input("Enter Median Income: "))

new_data = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income]
})

predicted_price = model.predict(new_data)

print(f"\nPredicted House Value: "f"{predicted_price[0]:.2f}")
plt.figure(figsize=(6, 6))
plt.scatter(y_test,y_pred,color='purple')

plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color='red',linewidth=2)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.savefig("actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.show()

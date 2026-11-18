import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset into a DataFrame
df = pd.read_csv('temp-lemon.csv')
df.dropna(how=all)
df.drop_duplicates()

# 2. Preprocess and extract features and target
# 'Temp' is assigned to X and 'Lemonde' to Y
X = df['Temp'].values
Y = df['Lemonde'].values

# 3. Add a bias term (column of ones) to X to account for the intercept
X_b = np.c_[np.ones((len(X), 1)), X]

# 4. Apply the Normal Equation to calculate the optimal parameters -𝜃=(𝑋.𝑇.𝑋)−1 . 𝑋.𝑇𝑦

theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(Y)

intercept = theta[0]
slope = theta[1]

print(f"Intercept (theta_0): {intercept}")
print(f"Slope (theta_1): {slope}")

# 5. Generate predictions for the fitted line
X_new = np.array([[df['Temp'].min()], [df['Temp'].max()]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
Y_predict = X_new_b.dot(theta)

# 6. Visualize the data and the fitted line
plt.scatter(X, Y, color='blue', alpha=0.1, label='Data points')
plt.plot(X_new, Y_predict, color='red', linewidth=2, label='Fitted Line')
plt.xlabel('Temperature')
plt.ylabel('Lemonade Sales')
plt.title('Linear Regression using Normal Equation')
plt.legend()
plt.tight_layout()

# Save the visualization
plt.savefig('fitted_line.png')
plt.show()
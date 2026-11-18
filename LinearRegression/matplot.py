import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('salary_dataset.csv')

x=df['YOE']
y=df['Salary']

plt.scatter(x,y,color="blue")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

plt.savefig('salary_plot.png')

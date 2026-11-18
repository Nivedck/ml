import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("salary_dataset.csv")

print(df.info())
df.dropna()

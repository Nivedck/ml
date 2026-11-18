import pandas as pd
import numpy as np 
import math
import random

df = pd.read_csv("iris.csv")

X = df[['sepal_length','sepal_width','petal_length','petal_width']].values.tolist()
y = df['species'].tolist()

dataset = list(zip(X,y))

random.seed(42)
random.shuffle(dataset)
split_index = int(0.7*len(df))
X_train = dataset[:split_index]
X_test  = dataset[split_index:]

def distance(p1,p2):
    total = 0
    for i in range(len(p1)):
        dist = p1[i]-p2[i]
        total += dist**2
    return math.sqrt(total)

#print(distance(X_train[0][0],X_train[1][0]))

def get_neighbours(point,k):
    neighbours = []
    for x,label in X_train:
        dis = distance(x,point)
        neighbours.append((dis,label))
    neighbours.sort()
    return neighbours[:k]

def predict(point,k):
    neighbours = get_neighbours(point,k)
    count={}
    for dis,label in neighbours:
        if label in count:
            count[label] += 1
        else:
            count[label] =1
    highest=-1
    new_label = None

    for label,count in count.items():
        if(count>highest):
            highest=count
            new_label=label
    return new_label




new_point = [5.5,2.4,3.7,1.0]

k = 5
result = predict(new_point, k)

print("New Point:", new_point)
print("Predicted Species:", result)
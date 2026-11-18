
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('salary_dataset.csv')

x_train = df['YOE'].tolist()
y_train = df['Salary'].tolist()


def cost_function(x,y,w,b):
    m = len(x)

    cost_sum = 0

    for i in range(m):
        y_pred = w * x[i] + b
        cost = (y[i]-y_pred) ** 2
        cost_sum+=cost

    total_cost = (1/(2*m)) * cost_sum

    return total_cost

def gradient_function(x,y,w,b):
    m=len(x)

    dc_dw = 0
    dc_db = 0

    for i in range(m):
        f = w * x[i]+b
        dc_dw += (f-y[i])*x[i]
        dc_db += (f-y[i])

    dc_db = (1/m) * dc_db
    dc_dw = (1/m) * dc_dw

    return dc_db, dc_dw


def gradient_descent(x,y,lr,iterations):
    w=0
    b=0

    for i in range(iterations):
        dc_db,dc_dw = gradient_function(x,y,w,b)

        w = w-lr* dc_dw
        b = b-lr* dc_db

        print(f"Iteration {i}: cost: {cost_function(x,y,w,b)}")

    return w,b

lr = 0.00001
iteration = 1000

final_w , final_b = gradient_descent(x_train,y_train,lr,iteration)
print(final_w)
print(final_b)

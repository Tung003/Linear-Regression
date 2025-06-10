from Linear_Regression import linear_regression, Gadient_Descent_Linear_Regression
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

file_path= "/home/chu-tung/Desktop/machine_learning/src/Linear-Regression/data/train.csv"
data_points=pd.read_csv(file_path)

x=data_points.drop("y",axis=1)
y=data_points.drop("x",axis=1)
x=np.array(x)
y=np.array(y)

Normal_equation_LR=linear_regression(x,y)
Normal_equation_LR.plot_fx(3)
print(Normal_equation_LR.find_fx())

Gadient_descent_LR=Gadient_Descent_Linear_Regression(x,y,0.00001,1000)
Gadient_descent_LR.plot_line_GD(3)


fig,ax=plt.subplots(figsize=(5,5))
ax.scatter(x,y,color="g")
ax.set_title("data points")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(["data point"])
plt.show()


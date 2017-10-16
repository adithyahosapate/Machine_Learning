import sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('data_regression.csv')
lr=LinearRegression()
lr.fit(df['col1'].reshape(-1,1),df['col2'].reshape(-1,1))
m=lr.coef_[0]
b=lr.intercept_
x=np.linspace(0,max(df['col1']),10000)


y=m*x+b
plt.scatter(df['col1'],df['col2'])

plt.plot(x,y)
plt.show()

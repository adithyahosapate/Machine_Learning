from __future__ import division
import sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

lr=LogisticRegression()
LR=LogisticRegression()


def model(x,m,b):
	return 1/(1+np.exp(-np.dot(m,x)-b))

# single feature
x_1=np.linspace(-1,1,10).reshape(-1,1)

y_1=np.array([1,1,1,1,1,0,0,0,0,0])


lr.fit(x_1,y_1)


#############################################################
#	plotting

for i,point in enumerate(x_1):
	plt.scatter(point,y_1[i])



x_vals=np.linspace(-4,4,1000)


y_vals=model(np.array([x_vals]),lr.coef_,lr.intercept_)[0].T
plt.plot(x_vals,y_vals)

plt.show()

#############################################################

#8 features

df=pd.read_csv('output.csv')
y=df['type'].reshape(-1)


val_list=[]

for key,val in df.items():
	if key!='type':
		val_list.append(val)
X=np.vstack(val_list)
print(X.T)
X=X.T



LR.fit(X,y)

#parameters
print(LR.coef_)
print(LR.intercept_)


#test
print(model(np.array([X[0]]).T,LR.coef_,LR.intercept_))
print(LR.predict(np.array([X[0]])))
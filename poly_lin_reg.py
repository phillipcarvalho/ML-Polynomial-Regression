#datapreprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('Position_Salariess.csv')
#X=df.iloc[:,1].values
#y=df.iloc[:,2].values
X=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values
#print(X)
#print(y)

#linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X, y)
#training with polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
X_poly=poly_reg.fit_transform(X)
lin_reg_2=LinearRegression()
lin_reg_2.fit(X_poly,y)

#Visualising linear 
plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualising polynomial
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(X_poly),color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#visualising polynomial more smoothly
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#predicting new test result
print(regressor.predict([[6.5]]))#linear
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))#polynomial
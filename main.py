import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values
Y = np.array(Y).reshape(-1,1)
X_train, X_test, Y_train , Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X_train)
print(Y_train)
print(X_test)
print(Y_test)

lr = LinearRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)

plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, lr.predict(X_train), color='blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Training Graph')
plt.show()

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, lr.predict(X_train), color='blue')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Test Graph')
plt.show()

print('Coefficients: ',lr.coef_)
print('Intercept: ',lr.intercept_)

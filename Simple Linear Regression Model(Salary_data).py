import numpy as np 	
import matplotlib.pyplot as plt
import pandas as pd	
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load the dataset
dataset = pd.read_csv(r'D:\Data Science & AI class note\27th Dec\Salary_Data.csv')

# Split the data into independent and dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values 

# Split the dataset into training and testing sets (80-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict the test set
y_pred = regressor.predict(X_test)

# Visualize the training set
plt.scatter(X_train, y_train, color='red') 
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

print(f"Intercept: {regressor.intercept_}")
print(f"coefficient: {regressor.coef_}")

comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison)

#Staistic for ML

dataset.mean()

dataset['Salary'].mean()

dataset.median()

dataset['Salary'].mode()

dataset.var()

dataset.std()

from scipy.stats import variation
variation(dataset.values)

dataset.corr()

dataset['Salary'].corr(dataset['YearsExperience'])

dataset.skew()

dataset['Salary'].skew()

dataset.sem()

#Z-score

import scipy.stats as stats
dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])

a = dataset.shape[0]
b = dataset.shape[1]

degree_of_freedom = a-b

#SSR
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

#SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

#SST
mean_total = np.mean(dataset.values)
SST = np.mean((dataset.values-mean_total)**2)
print(SST)

#r2
r_square = 1-SSR/SST
print(r_square)

import pickle
filename = 'regressor.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled saved as Linear_regression_model.pkl")

import os
print(os.getcwd())

bias = regressor.score(X_train,y_train)
print(bias)

variance = regressor.score(X_test,y_test)
print(variance)






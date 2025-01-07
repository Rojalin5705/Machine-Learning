import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv(r"D:\Data Science & AI class note\26th Dec\Data.csv")

X = dataset.iloc[:,:-1].values

y = dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0])

X[:,0] = labelencoder_X.fit_transform(X[:,0])

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state=0) 

#Feature Scaling(06/01/2025)

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)                            
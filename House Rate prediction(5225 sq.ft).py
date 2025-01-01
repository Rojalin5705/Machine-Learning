# Step_1 Import the Libraries 

# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

#step_2 load the daatset

df = pd.read_csv(r'D:\Data Science & AI class note\28th Dec\26th- SLR\26th- SLR\SLR - House price prediction\House_data.csv')
print("")
df.head()

df.describe()

df.dtypes

df.duplicated().any()

df.head()

# we see date & Id are not important here so drop the columns

x = df[['sqft_living']].values  # Independent variable

print(x)

y = df['price'].values  # Dependent variable

print(y)

# train the data 


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

# predcit the test


regressor = LinearRegression()

regressor.fit(x_train,y_train)

# Predcit of the test & training set result

y_train_pred = regressor.predict(x_train)
y_test_pred = regressor.predict(x_test)

# Plotting the training data and regression line
plt.figure(figsize=(10, 6))
plt.scatter(x_train,y_train,color='green',label='Training Data')
plt.plot(x_train,y_train_pred,color='red', label='Regression Line (Training)')
plt.show()

# Adding labels and title
plt.grid(True)
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('Linear Regression (Train Split)')
plt.legend()
plt.show()

# Plotting the testing data and regression line

plt.scatter(x_test, y_test, color='red', label='Testing Data')
plt.plot(x_test,y_test_pred, color='orange', label='Regression Line (Testing)')
plt.show()


# Adding labels and title
plt.xlabel('quare Feet')
plt.ylabel('Price')
plt.title('Linear Regression (Test Split)')
plt.legend()
plt.show()

# Print model performance on both training and testing sets

print(f"regressor Performance on Training Data: {regressor.score(x_train, y_train)}")
print(f"regressor Performance on Testing Data: {regressor.score(x_test, y_test)}")

# predict the future value 

new_value = regressor.predict([[5225]])

print(new_value)

# y = mx + c

print(regressor.coef_[0])
print(regressor.intercept_)

y = 284.1477103801124 *5225 -48536.690058289096

print(y)


# Plot the data points and regression line

# Evaluate model performance 

r2 = r2_score(y_test, y_test_pred)
print(r2)



# Plotting the training data and regression line
# Highlight the future prediction
future_sqft = 5225
future_price  = regressor.predict([[5225]])

plt.figure(figsize=(10, 6))
plt.scatter(x_train,y_train,color='green',label='Training Data')
plt.plot(x_train,y_train_pred,color='red', label='Regression Line (Training)')
plt.show()


# Highlight the future prediction

plt.scatter(future_sqft,future_price, color='blue',label="Future Value (5225 sq ft)")
plt.axvline(future_sqft,color='orange', linestyle='--', label="5225 sq ft")
plt.axhline(future_price, color='blue', linestyle='--', label="future_price")
plt.show()


# Adding labels and title
# Labels and title
plt.title("Simple Linear Regression: Training Set with Future Prediction", fontsize=14)
plt.xlabel("Square Footage of Living Area", fontsize=12)
plt.ylabel("Price", fontsize=12)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("The future value is:",future_price)



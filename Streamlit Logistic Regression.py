import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load the dataset
dataset = pd.read_csv(r"D:\Data Science & AI class note\20th Jan\logit classification.csv")

# Features and target variable
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train the logistic regression model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predict the test set results
y_pred = classifier.predict(X_test)

# Evaluate the model
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Streamlit UI components
st.title("Vehicle Purchase Prediction using Logistic Regression")

st.write("This application predicts whether a user is likely to purchase a vehicle based on age and estimated salary.")

# Display confusion matrix and accuracy score
st.subheader("Model Evaluation")
st.write(f"Confusion Matrix: \n{cm}")
st.write(f"Accuracy Score: {ac}")
st.write(f"Classification Report: \n{cr}")

# Input fields for prediction
age = st.number_input("Enter Age", min_value=18, max_value=100, step=1)
estimated_salary = st.number_input("Enter Estimated Salary: ", min_value=0, step=1000)

# Prediction on user input
if st.button("Predict"):
    # Prepare input data
    input_data = np.array([[age, estimated_salary]])
    input_data_scaled = sc.transform(input_data)  # scale the input data
    
    # Predict the result
    prediction = classifier.predict(input_data_scaled)
    
    # Display prediction result
    if prediction[0] == 1:
        st.success("The user is likely to purchase the vehicle.")
    else:
        st.error("The user is unlikely to purchase the vehicle.")

# Saving future predictions
if st.button("Predict Future Sales"):
    # Prepare dataset for prediction
    dataset1 = pd.read_csv(r"D:\Data Science & AI class note\20th Jan\logit classification.csv")
    d2 = dataset1.copy()
    dataset1 = dataset1.iloc[:, [2, 3]].values

    # Scale the dataset
    M = sc.transform(dataset1)

    # Predict future sales
    y_pred1 = classifier.predict(M)

    # Add predictions to the dataset
    d2['y_pred1'] = y_pred1

    # Save the predictions to a new CSV file
    d2.to_csv('Logistic_Regression_Prediction_Car_Sales.csv', index=False)

    st.write("Future predictions have been saved to 'Logistic_Regression_Prediction_Car_Sales.csv'.")


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Streamlit title
st.title("K-NN Classifier for Social Network Ads")

st.write("""
This app implements a K-NN classifier to predict social network ad clicks. Upload a CSV dataset to run the model and see the results!
""")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded dataset
    dataset = pd.read_csv(uploaded_file)
    
    st.write("Dataset Preview:")
    st.dataframe(dataset.head())  # Show first few rows of the dataset

    # Extracting the feature columns and target variable
    X = dataset.iloc[:, [2, 3]].values  # Assuming columns 2 and 3 are the features
    y = dataset.iloc[:, -1].values      # Assuming the last column is the target variable

    # Splitting the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Training the K-NN model
    classifier = KNeighborsClassifier(n_neighbors=4, p=1)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Display results
    st.write("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    # Model Accuracy
    ac = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy Score: {ac:.2f}")

    # Classification Report
    cr = classification_report(y_test, y_pred)
    st.write("Classification Report:")
    st.text(cr)

    # Bias and Variance
    bias = classifier.score(X_train, y_train)
    variance = classifier.score(X_test, y_test)
    st.write(f"Bias (Training Accuracy): {bias:.2f}")
    st.write(f"Variance (Testing Accuracy): {variance:.2f}")

    # Optional: Visualizing the decision boundary
    st.write("Visualizing Decision Boundary:")
    fig, ax = plt.subplots()
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
    ax.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                alpha=0.75, cmap='Blues')
    ax.scatter(X_set[:, 0], X_set[:, 1], c=y_set, s=25, edgecolor='k', cmap='coolwarm')
    ax.set_title("K-NN Classifier (Training set)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    st.pyplot(fig)

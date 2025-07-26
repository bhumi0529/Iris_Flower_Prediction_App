import streamlit as st
import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load and display image
image = Image.open('iris_image.jpg')  # Make sure this image exists
st.image(image, caption='Iris Flower Types', use_container_width=True)

# Load the trained model
model = joblib.load("iris_model.pkl")

# App title
st.title("🌸 Iris Flower Prediction App")
st.write("Enter flower measurements below:")

# User input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Predict when button clicked
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Make prediction
    prediction = model.predict(features)

    # Feature importance chart
    importances = model.feature_importances_
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']

    fig, ax = plt.subplots()
    ax.barh(feature_names, importances, color='skyblue')
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importances")

    st.pyplot(fig)

    # Class names
    iris_classes = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"🌼 Predicted Species: **{iris_classes[prediction[0]]}**")

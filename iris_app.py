import streamlit as st
import joblib
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import base64

def set_background(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("back.jpg")  
st.markdown("<h1 style='text-align: center; color: black;'>🌸 Iris Flower Prediction App 🌸</h1>", unsafe_allow_html=True)


# Load and display image
image = Image.open('flower.jpeg')  
st.image(image, caption='Iris Flower Types', use_container_width=True)

# Load the trained model
model = joblib.load("iris_model.pkl")

# App title
st.markdown(
    """
    <style>
    .stSlider > div > div > div > label {
        font-size: 30px important;
        font-weight: bold;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Slider label text (feature names) */
    .stSlider > div > div > div > label {
        font-size: 30px important;
        font-weight: bolder;
        color: black;
    }

    /* Slider numbers (tick values) */
    .stSlider span {
        font-size: 28px important;
        font-weight: bolder;
        color: black important;
    }

    /* Description text before sliders */
    .stMarkdown p {
        font-size: 18px;
        font-weight: 500;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
"""
<style>
/* Slider label and tick value styling */
.css-1cpxqw2, .css-1hynsf2, .css-qrbaxs, .css-1y4p8pa, .css-14xtw13 {
    font-size: 22px !important;
    font-weight: bold !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)


st.write("Enter flower measurements below:")

st.subheader("🌼 Input Features")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
with col2:
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
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
    st.markdown(
    f"""
    <div style="
        background-color: #dff0d8;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        color: #1a1a1a;
        font-size: 18px;
        font-weight: bold;">
        🌼 Predicted Species: <span style="color:#000;">{iris_classes[prediction[0]]}</span>
    </div>
    """,
    unsafe_allow_html=True
)


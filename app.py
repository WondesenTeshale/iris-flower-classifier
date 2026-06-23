import streamlit as st
import joblib
import numpy as np

model = joblib.load("iris_model.pkl")

st.title("Iris Flower Classifier")

st.write("Enter flower measurements")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.2)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.3)

if st.button("Predict"):
    features = np.array([
        [sepal_length, sepal_width,
         petal_length, petal_width]
    ])

    prediction = model.predict(features)[0]

    species = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    st.success(
        f"Prediction: {species[prediction]}"
    )
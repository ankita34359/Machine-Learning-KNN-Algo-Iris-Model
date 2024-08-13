import streamlit as st
import joblib 
import numpy as np

st.header("Iris Flower Classification")

sepal_length = st.number_input("Enter Sepal Length")
sepal_width = st.number_input("Enter Sepal Width")
petal_length = st.number_input("Enter Petal Length")
petal_width = st.number_input("Enter petal Width")

button = st.button("Submit")

loaded_model = joblib.load(r"C:\Users\Lenovo\OneDrive\Desktop\Juypter_projects\Iris_model.pkl")

X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

predicted_value = loaded_model.predict(X)

decode_dict = {0:"Iris-Setosa", 1:"Iris-Virginica", 2:"Iris-Versiclor"}

predicted_name = decode_dict[predicted_value[0]]

if button:
    st.info(predicted_value)
    st.info(predicted_name)

    
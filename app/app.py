import joblib
import pandas as pd
import streamlit as st

@st.cache_resource # loading the model only one time to prevent loading it each time the user clicks the button 
def load_model():
    return joblib.load("../models/rf_model.joblib")

model = load_model()

names = {0: "VIP", 1: "Inactive", 2: "Regular"}

st.title("CartSense")
recency = st.number_input("Recency", step=10)
frequency = st.number_input("Frequency", step=1)
monetary = st.number_input("Monetary", step=100)


if st.button("CHECK"):
    customer = pd.DataFrame({
        "Recency" : [recency],
        "Frequency" : [frequency],
        "Monetary" : [monetary],
    })
    cluster = model.predict(customer)[0]
    st.success(names[cluster])
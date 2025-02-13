import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib as joblib


insurance_df = pd.read_csv("insurance.csv")

st.title("Insurance Prediction App")
st.dataframe(insurance_df.head())


#Load the trained ML Model
with open("Predict_Insurance_Cost_MLmodel.pickle", "rb") as f:
    model = pickle.load(f)

#Streamlit App
st.title("Streamlit Prediction App")

#Dropdown menu for selection

age = st.selectbox("Age", list(range(18, 64)))
diabetes = 1 if st.selectbox("Diabetes", ["No","Yes"]) == "Yes" else 0
bp = 1 if st.selectbox("BP", ["No","Yes"]) == "Yes" else 0
transplant = 1 if st.selectbox("AnyTransplants", ["No","Yes"])  == "Yes" else 0
chronicdiseases = 1 if st.selectbox("BPAnyChronicDiseases", ["No","Yes"])  == "Yes" else 0
height = st.number_input("Height", min_value=140, max_value=200)
weight = st.number_input("Weight", min_value=40, max_value=250)
allergies = 1 if st.selectbox("KnownAllergies", ["No","Yes"])  == "Yes" else 0
history = 1 if st.selectbox("HistoryOfCancerInFamily", ["No","Yes"])  == "Yes" else 0
surgery = st.number_input("NumberOfMajorSurgeries", min_value=0, max_value=10)


if st.button("Get Price Prediction"):
    #predict here

    input_insurance = [age, diabetes, bp, transplant, chronicdiseases, height, weight, allergies, history, surgery]
    price = model.predict([input_insurance])[0]

    st.header(str(round(price, 0)))
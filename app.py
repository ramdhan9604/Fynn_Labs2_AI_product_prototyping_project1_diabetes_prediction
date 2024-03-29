import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

model = open('model.pkl', 'rb')
classifier = pickle.load(model)


st.sidebar.header('Diabetes Prediction')

st.title('Diabetes Prediction(Above 21 Years of Age)')


name = st.text_input("Name:")

glucose = st.number_input("Plasma Glucose Concentration :")
st.markdown('Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')

bp =  st.number_input("Diastolic blood pressure (mm Hg):")
st.markdown('BloodPressure: Diastolic blood pressure (mm Hg)')

skin = st.number_input("Triceps skin fold thickness (mm):")
st.markdown('SkinThickness: Triceps skin fold thickness (mm)')

insulin = st.number_input("2-Hour serum insulin (mu U/ml):")
st.markdown('Insulin: 2-Hour serum insulin (mu U/ml)')


bmi = st.number_input("Body mass index (weight in kg/(height in m)^2):")
st.markdown('BMI: Body mass index (weight in kg/(height in m)^2)')

dpf = st.number_input("Diabetes Pedigree Function:")
st.markdown('DiabetesPedigreeFunction: Diabetes pedigree function')


age = st.number_input("Age:")
st.markdown('Age: Age (years)')


submit = st.button('Predict')
st.markdown('Outcome: Class variable (0 or 1)')


if submit:
    prediction = classifier.predict([[glucose, bp, skin, insulin, bmi, dpf, age]])
    if prediction == 0:
        st.write('Congratulation!',name,'You are not diabetic')
    else:
        st.write(name,", we are really sorry to say but it seems like you are Diabetic. But don't lose hope, we have suggestions for you:")
        st.markdown('[Visit Here](https://www.mayoclinic.org/diseases-conditions/type-2-diabetes/in-depth/diabetes-prevention/art-20047639#:~:text=Diabetes%20prevention%3A%205%20tips%20for%20taking%20control%201,Skip%20fad%20diets%20and%20make%20healthier%20choices%20)')



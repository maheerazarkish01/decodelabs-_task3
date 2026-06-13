import streamlit as st
import pickle
import numpy as np


# Load model

model = pickle.load(
    open("heart_model.pkl","rb")
)


scaler = pickle.load(
    open("scaler.pkl","rb")
)



st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)


st.title(
"❤️ Heart Disease Prediction System"
)


st.write(
"Machine Learning based system to predict heart disease risk"
)



st.sidebar.header(
"Patient Information"
)



age = st.sidebar.number_input(
"Age",
20,
100
)



sex = st.sidebar.selectbox(
"Gender",
["Male","Female"]
)


cp = st.sidebar.number_input(
"Chest Pain Type",
0,
3
)


trestbps = st.sidebar.number_input(
"Blood Pressure"
)


chol = st.sidebar.number_input(
"Cholesterol"
)


fbs = st.sidebar.selectbox(
"Fasting Blood Sugar",
[0,1]
)


restecg = st.sidebar.number_input(
"ECG Result",
0,
2
)


thalach = st.sidebar.number_input(
"Maximum Heart Rate"
)


exang = st.sidebar.selectbox(
"Exercise Pain",
[0,1]
)


oldpeak = st.sidebar.number_input(
"ST Depression"
)



if st.button("Predict Heart Disease"):


    if sex=="Male":
        sex=1
    else:
        sex=0



    input_data=np.array(
    [[
    age,
    sex,
    cp,
    trestbps,
    chol,
    fbs,
    restecg,
    thalach,
    exang,
    oldpeak,
    0,
    0,
    0
    ]]
    )


    input_scaled=scaler.transform(
        input_data
    )


    prediction=model.predict(
        input_scaled
    )


    if prediction[0]==1:

        st.error(
        "⚠️ High Risk of Heart Disease"
        )


    else:

        st.success(
        "✅ Low Risk of Heart Disease"
        )
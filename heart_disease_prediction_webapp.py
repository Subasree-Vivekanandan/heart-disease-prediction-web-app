# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 11:08:10 2023

@author: HP
"""

import numpy as np
import pickle 
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

loaded_model = pickle.load(open('C:/Users/HP/Notebook files/trained_hd_model.sav','rb'))

def heartdisease_prediction(input_data):

    input_data_as_numpy_Array = np.array(input_data)

    input_reshaped = input_data_as_numpy_Array.reshape(1,-1)

    prediction = loaded_model.predict(input_reshaped)
    if(prediction[0]==0):
        return "The person does not have heart disease"
    else:
        return "The person has heart disease"
    
    
#input_data = (63,0,0,124,197,0,1,136,1,0.0,1,0,2)


def main():
    
    # giving a title 
    st.title("Heart disease prediction web app")
    age = st.text_input("Enter age as number")
    sex = st.text_input("Enter gender (1-Male ,0-Female")
    cp  = st.text_input("Enter Chest paint type (0-3)")
    trestbps = st.text_input("Resting blood pressure")
    chol = st.text_input("Enter serum cholesterol in mg/dl")
    fbs = st.text_input("Enter fasting blood sugar")
    restecg = st.text_input("Resting electrocardiographic results")
    thalach = st.text_input("Maximum heart rate achieved")
    exang = st.text_input("Exercise induced angina (1-yes,0-no)")
    oldpeak = st.text_input("ST depression induced by exercise relative to rest")
    slope = st.text_input("slope of the peak exercise ST segment")
    ca = st.text_input("number of major vessels (0-3) colored by flourosopy")
    thal = st.text_input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")
    
    # code for prediction
    diagnosis = ''
    
    # creating a button
    if st.button('HD Test Result'):
        diagnosis = heartdisease_prediction([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thalach])
    
    st.success(diagnosis)
        
if __name__ == '__main__':
    main()
    
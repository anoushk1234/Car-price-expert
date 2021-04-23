import numpy as np
import streamlit as st
import pandas as pd
import pickle
from streamlit.config import on_config_parsed


pkl_file = open("car_price_model.pkl", "rb")
model = pickle.load(pkl_file)
Fuel_Type_Diesel = 0
st.title("Want to know the your car's resale?🚗")

html_title = '''

    <style>
        h3{
           background-color: blueviolet;
            border:1px solid blueviolet;
            border-radius: 4px;
            padding:10px 5px;
            font-size: large;
        width:59%
        }
    </style>
    <h3>
        Use this ML calculator to generate the best price
    </h3>

'''
st.markdown(html_title, unsafe_allow_html=True)


yr = st.text_input("Car Year", "0")
yr_variant = 2021-int(yr)

kms_driven = np.log(float(st.text_input("Kms Driven", "0")))

owner = int(st.text_input("Owner(s)", "0"))

Transmission_Manual = st.selectbox(
    "Transimission",
    ('Manual', 'Automatic'), key="1")
if Transmission_Manual == 'Manual':
    Transmission_Manual = 1
else:
    Transmission_Manual = 0


Fuel_Type_Petrol = st.selectbox(
    "Fuel Type",
    ('Petrol', 'Diesel',"CNG"), key="2")
if Fuel_Type_Petrol == "Petrol":
    Fuel_Type_Petrol = 1
    Fuel_Type_Diesel = 0
elif Fuel_Type_Petrol== "Diesel":
    Fuel_Type_Petrol = 0
    Fuel_Type_Diesel = 1
else:
    Fuel_Type_Petrol = 0
    Fuel_Type_Diesel = 0


Seller_Type_Individual = st.selectbox(
    "Seller Type",
    ('Individual', 'Dealer'), key="3")
if Seller_Type_Individual == 'Individual':
    Seller_Type_Individual = 1
else:
    Seller_Type_Individual = 0

output=0
present_price = float(st.text_input("Purchase Price", "0"))
if st.button("Predict"):
    prediction=model.predict([[present_price,kms_driven,owner,yr_variant,Fuel_Type_Diesel,Fuel_Type_Petrol,Seller_Type_Individual,Transmission_Manual]])
    output=round(prediction[0],2)

if output<0:
    st.write("You can't sell this car")
elif output>0:
    st.success("You can sell this car at {}".format(output))


html_links='''
<style>
 a{
     color:white;
 }
</style>
<div>
<a href="https://twitter.com/anoushk77"><u>Twitter<u></a>
<a href="https://github.com/anoushk1234"><u>Github<u></a>
</div>
'''
st.markdown(html_links,unsafe_allow_html=True)
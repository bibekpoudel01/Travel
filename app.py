import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import numpy as np
import streamlit as st
import pickle

import pickle
with open('random_forest_model.pkl', 'rb') as file:
    Random_Forest=pickle.load(file)
with open('xgb_model.pkl', 'rb') as file:
    XGBClassifier=pickle.load(file)
with open('preprocessor.pkl', 'rb') as file:
    preprocessor=pickle.load(file)

import streamlit as st
import pandas as pd

st.title("Travel Product Prediction")
TypeofContact = st.selectbox('Type of Contact', ['Self Enquiry', 'Company Invited'])
Occupation = st.selectbox('Occupation', ['Large Business', 'Free Lancer','Small Business', 'Salaried'])
Gender = st.selectbox('Gender', ['Male', 'Female'])
ProductPitched = st.selectbox('Product Pitched', ['Basic','Deluxe', 'Standard', 'Super Deluxe','King'])
MaritalStatus = st.selectbox('Marital Status', ['Married', 'Unmarried', 'Divorced'])
Designation = st.selectbox('Designation', ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
Age = st.slider('Age', 18, 70, 38)
CityTier = st.slider('City Tier', 1, 3, 1)
DurationOfPitch = st.slider('Duration of Pitch', 1, 30, 9)
NumberOfFollowups = st.slider('Number of Followups', 0, 10, 3)
PreferredPropertyStar = st.slider('Preferred Property Star', 1, 5, 3)
NumberOfTrips = st.slider('Number of Trips', 0, 20, 4)
Passport = st.selectbox('Passport', [0, 1])
PitchSatisfactionScore = st.slider('Pitch Satisfaction Score', 1, 5, 5)
OwnCar = st.selectbox('Own Car', [0, 1])
MonthlyIncome = st.number_input('Monthly Income', 0, 100000, 1000)
TotalVisiting = st.slider('Total Visiting', 0, 20, 2)

# Convert to DataFrame for model input
input_dict = {
    'Age': [Age],
    'TypeofContact': [TypeofContact],
    'CityTier': [CityTier],
    'DurationOfPitch': [DurationOfPitch],
    'Occupation': [Occupation],
    'Gender': [Gender],
    'NumberOfFollowups': [NumberOfFollowups],
    'ProductPitched': [ProductPitched],
    'PreferredPropertyStar': [PreferredPropertyStar],
    'MaritalStatus': [MaritalStatus],
    'NumberOfTrips': [NumberOfTrips],
    'Passport': [Passport],
    'PitchSatisfactionScore': [PitchSatisfactionScore],
    'OwnCar': [OwnCar],
    'Designation': [Designation],
    'MonthlyIncome': [MonthlyIncome],
    'TotalVisiting': [TotalVisiting]
}



input_df = pd.DataFrame(input_dict)
X_input_transformed = preprocessor.transform(input_df)
rf_pred = Random_Forest.predict(X_input_transformed)
rf_pred_proba =Random_Forest.predict_proba(X_input_transformed)[:, 1]
st.write("Random Forest Prediction:", rf_pred[0])
st.write("Random Forest Probability:", rf_pred_proba[0])
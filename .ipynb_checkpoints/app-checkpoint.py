import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import numpy as np
import streamlit as st
import pickle

with open('random_forest_model.pkl','wb') as f:
    pickle.dump(models['Random Forest'], f)

with open('xgb_model.pkl','wb') as f:
    pickle.dump(models['XGBClassifier'], f)

with open("preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

import streamlit as st
import pandas as pd

st.title("Travel Product Prediction")
TypeofContact = st.selectbox('Type of Contact', ['Self Enquiry', 'Agent Enquiry'])
Occupation = st.selectbox('Occupation', ['Large Business', 'Small Business', 'Salaried', 'Student'])
Gender = st.selectbox('Gender', ['Male', 'Female'])
ProductPitched = st.selectbox('Product Pitched', ['Deluxe', 'Standard', 'Budget'])
MaritalStatus = st.selectbox('Marital Status', ['Married', 'Unmarried'])
Designation = st.selectbox('Designation', ['Manager', 'Executive', 'Assistant'])
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
rf_pred = rf_model.predict(X_input_transformed)
rf_pred_proba = rf_model.predict_proba(X_input_transformed)[:, 1]
st.write("Random Forest Prediction:", rf_pred[0])
st.write("Random Forest Probability:", rf_pred_proba[0])
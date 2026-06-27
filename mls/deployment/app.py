import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# Safely catch token if available for private/gated tracking
hf_token = os.getenv("HF_TOKEN")

# Explicitly add repo_type="dataset" to point to where your files live
model_path = hf_hub_download(
    repo_id="Shamsul26/MLOPS", 
    filename="best_tourism_model_v1.joblib",
    repo_type="dataset",
    token=hf_token
)
model = joblib.load(model_path)

# Streamlit UI for Tourism Package
st.title("Tourism Package Prediction")
st.write("Predicting the likelihood of purchasing the Wellness Tourism Package.")

# User inputs...
age = st.number_input("Age", min_value=18, max_value=100, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
CityTier = st.selectbox("City Tier", ["1", "2", "3"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Occupation = st.selectbox("Occupation", ["Freelancer", "Large Business","Small Business","Salaried"])
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=5, value=2)
NumberOfFollowups = st.number_input("No of Followups", min_value=1, max_value=10, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", ["3", "4", "5"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single"])
NumberOfTrips = st.number_input("Number of Trips", min_value=1, max_value=22, value=3)
Passport = st.selectbox("Passport", ["Yes", "No"])
OwnCar = st.selectbox("Own Car", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=3, value=0)
Designation = st.selectbox("Designation", ["Executive", "AVP","Manager","Senior Manager","VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=50000)
DurationOfPitch = st.number_input("Pitch Duration", min_value=5 , max_value=200, value=5)
Productpitched = st.selectbox("Product Pitched", ["Basic", "Delux", "King","Standard","Super Delux"])
PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", ["1","2","3","4","5"])

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': TypeofContact,
    'CityTier': int(CityTier),
    'Gender': Gender,
    'Occupation': Occupation,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': int(PreferredPropertyStar),
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'ProductPitched': Productpitched,
    'DurationOfPitch': DurationOfPitch,
    'PitchSatisfactionScore': int(PitchSatisfactionScore)
}])

if st.button("Predict Purchasing"):
    prediction = model.predict(input_data)[0]
    result = "Customer WILL purchase the package" if prediction == 1 else "Customer WILL NOT purchase the package"

    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"Result: **{result}**")
    else:
        st.warning(f"Result: **{result}**")

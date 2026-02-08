import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')    

# Load the encoders and scaler
with open('one_hot_encoder.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)        

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)  

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Streamlit app
st.title("Customer Churn Prediction")

# Input fields

CreditScore = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
Geography = st.selectbox('Geography', one_hot_encoder.categories_[0])
Gender = st.selectbox('Gender', label_encoder.classes_)
Age = st.slider('Age', 18, 92)
Tenure = st.slider('Tenure', 0, 10)
Balance = st.number_input('Balance')                
NumOfProducts = st.slider('Number of Products', 1, 4)
HasCrCard = st.selectbox('Has Credit Card', [0, 1])
IsActiveMember = st.selectbox('Is Active Member', [0, 1])
EstimatedSalary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

# Encode Gender
gender_encoded = label_encoder.transform([Gender])[0]

# One-hot encode Geography
geo_encoded = one_hot_encoder.transform([[Geography]])
geo_encoded_df = pd.DataFrame(
    geo_encoded,
    columns=one_hot_encoder.get_feature_names_out(['Geography'])
)

# Base numeric features (NO Geography here)
input_df = pd.DataFrame([{
    'CreditScore': CreditScore,
    'Gender': gender_encoded,
    'Age': Age,
    'Tenure': Tenure,
    'Balance': Balance,
    'NumOfProducts': NumOfProducts,
    'HasCrCard': HasCrCard,
    'IsActiveMember': IsActiveMember,
    'EstimatedSalary': EstimatedSalary
}])

# Combine numeric + encoded geography
final_input = pd.concat([input_df, geo_encoded_df], axis=1)

# Scale
final_input_scaled = scaler.transform(final_input)



# Predict
if st.button('Predict Churn'):
    prediction = model.predict(final_input_scaled)
    st.write(f"Churn Probability: {prediction[0][0]:.4f}")
    if prediction[0][0] > 0.5:
        st.success("The customer is likely to churn.")
    else:
        st.success("The customer is not likely to churn.")

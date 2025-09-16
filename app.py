import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("blackfriday_modelstr.pkl")

# Define input fields
st.title("🛍️ Black Friday Purchase Predictor")

gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
age = st.number_input("Age (encoded)", min_value=0, max_value=6, value=3)
occupation = st.number_input("Occupation", min_value=0, max_value=20, value=2)
city_category = st.number_input("City Category (encoded)", min_value=0, max_value=2, value=1)
stay_years = st.number_input("Stay in Current City (Years)", min_value=0, max_value=20, value=2)
marital_status = st.selectbox("Marital Status", [0, 1])
product_cat_1 = st.number_input("Product Category 1", min_value=1, max_value=20, value=3)
product_cat_2 = st.number_input("Product Category 2", min_value=-1, max_value=20, value=-1)
product_cat_3 = st.number_input("Product Category 3", min_value=-1, max_value=20, value=-1)

# Prepare input
input_df = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "Occupation": occupation,
    "City_Category": city_category,
    "Stay_In_Current_City_Years": stay_years,
    "Marital_Status": marital_status,
    "Product_Category_1": product_cat_1,
    "Product_Category_2": product_cat_2,
    "Product_Category_3": product_cat_3,
}])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"🛒 Predicted Purchase Amount: ₹{prediction:.2f}")

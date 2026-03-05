import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the Model and Scaler
try:
    model = joblib.load("xgb_best_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

st.title("Purchase Intent Prediction")

# ---- User Inputs ----
col1, col2 = st.columns(2)
with col1:
    price = st.number_input("Product Price ($)", min_value=0.0, value=500.0)
    age = st.number_input("Customer Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Customer Gender", ["Female", "Male"])

with col2:
    frequency = st.number_input("Purchase Frequency", min_value=1, value=2)
    category = st.selectbox("Category", ["Headphones", "Laptops", "Smart Watches", "Smartphones", "Tablets"])
    brand = st.selectbox("Brand", ["Apple", "HP", "Other Brands", "Samsung", "Sony"])

# ---- Prediction Logic ----
if st.button("Calculate Probability"):
    
    # 1. Create the base dataframe with ALL 16 features (initialize with 0)
    # This ensures we match the "Features expected by loaded_xgb_model" exactly.
    input_df = pd.DataFrame(columns=[
        'ProductPrice', 'CustomerAge', 'CustomerGender', 'PurchaseFrequency',
        'ProductCategory_Headphones', 'ProductCategory_Laptops', 
        'ProductCategory_Smart Watches', 'ProductCategory_Smartphones', 
        'ProductCategory_Tablets', 'ProductBrand_Apple', 'ProductBrand_HP', 
        'ProductBrand_Other Brands', 'ProductBrand_Samsung', 'ProductBrand_Sony', 
        'IsMostSoldBrand', 'Price Per Frequency'
    ])
    
    # 2. Fill in the user values
    new_row = {
        "ProductPrice": price,
        "CustomerAge": age,
        "CustomerGender": 1 if gender == "Male" else 0,
        "PurchaseFrequency": frequency,
        "IsMostSoldBrand": 1 if brand in ["Apple", "Samsung"] else 0,
        "Price Per Frequency": price / frequency if frequency > 0 else 0
    }
    
    # Set the specific Category and Brand to 1
    new_row[f"ProductCategory_{category}"] = 1
    new_row[f"ProductBrand_{brand}"] = 1
    
    # Add the row to the dataframe
    input_df = pd.concat([input_df, pd.DataFrame([new_row])], ignore_index=True).fillna(0)

    try:
        # 3. CRITICAL FIX: Scale ONLY the 4 columns the scaler knows
        scaler_cols = ['ProductPrice', 'CustomerAge', 'PurchaseFrequency', 'Price Per Frequency']
        
        # Scale only these 4 columns
        input_df[scaler_cols] = scaler.transform(input_df[scaler_cols])
        
        # 4. Predict using the fully prepared 16-column dataframe
        # We ensure the column order is exactly what the model expects
        model_order = [
            'ProductPrice', 'CustomerAge', 'CustomerGender', 'PurchaseFrequency',
            'ProductCategory_Headphones', 'ProductCategory_Laptops', 
            'ProductCategory_Smart Watches', 'ProductCategory_Smartphones', 
            'ProductCategory_Tablets', 'ProductBrand_Apple', 'ProductBrand_HP', 
            'ProductBrand_Other Brands', 'ProductBrand_Samsung', 'ProductBrand_Sony', 
            'IsMostSoldBrand', 'Price Per Frequency'
        ]
        
        final_input = input_df[model_order]
        
        # Predict
        prob = model.predict_proba(final_input)[0][1]
        
        st.divider()
        st.write(f"### Purchase Probability: {prob:.2%}")
        if prob > 0.5:
            st.success("Result: **Likely to Buy**")
        else:
            st.error("Result: **Unlikely to Buy**")
            
    except Exception as e:
        st.error(f"Prediction Error: {e}")
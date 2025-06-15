import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("Quikr_car.csv")
df = df[df['Price'] != 'Ask For Price']
df['Price'] = df['Price'].str.replace('₹', '').str.replace(',', '').astype(int)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df[df['Year'].notnull()]
df['Year'] = df['Year'].astype(int)
df = df[df['Kms_driven'].notnull()]
df['Kms_driven'] = df['Kms_driven'].str.replace(',', '').str.extract('(\d+)').astype(float)
df = df[df['Fuel_type'].notnull()]
df['Company'] = df['Name'].str.split().str[0]
df['Model'] = df['Name'].str.split().str[1]
df = df[['Company', 'Model', 'Year', 'Price', 'Kms_driven', 'Fuel_type']]
df.dropna(inplace=True)

# Train model
X = df[['Company', 'Model', 'Year', 'Kms_driven', 'Fuel_type']]
y = df['Price']
X_encoded = pd.get_dummies(X, drop_first=True)
model = LinearRegression()
model.fit(X_encoded, y)

# Utility functions
def compare_with_market(company, model_name, year, predicted_price):
    filtered = df[(df['Company'] == company) & (df['Model'] == model_name) & (df['Year'] == year)]
    if len(filtered) == 0:
        return "Market data not available"
    avg_market_price = filtered['Price'].mean()
    if predicted_price > avg_market_price * 1.10:
        return "Above Market Average"
    elif predicted_price < avg_market_price * 0.90:
        return "Below Market Average"
    else:
        return "At Market Rate"

def predict_resale_price(current_price, years=2, annual_depreciation_rate=0.15):
    return round(current_price * ((1 - annual_depreciation_rate) ** years), 2)

# Streamlit UI
st.title("Used Car Price Predictor")

company = st.selectbox("Select Car Company", sorted(df['Company'].unique()))
model_name = st.selectbox("Select Car Model", sorted(df[df['Company'] == company]['Model'].unique()))
year = st.selectbox("Select Year", sorted(df['Year'].unique(), reverse=True))
kms_driven = st.number_input("Enter Kilometers Driven", min_value=0, step=1000)
fuel_type = st.selectbox("Select Fuel Type", df['Fuel_type'].unique())
years_to_predict = st.slider("Years Ahead for Resale Value Prediction", 1, 10, 2)

if st.button("Predict Price"):
    input_df = pd.DataFrame([[company, model_name, year, kms_driven, fuel_type]],
                            columns=['Company', 'Model', 'Year', 'Kms_driven', 'Fuel_type'])
    input_encoded = pd.get_dummies(input_df, drop_first=True).reindex(columns=X_encoded.columns, fill_value=0)

    predicted_price = model.predict(input_encoded)[0]
    st.success(f"Predicted Current Price: ₹{int(predicted_price)}")

    comparison = compare_with_market(company, model_name, year, predicted_price)
    st.info(f"Market Comparison: {comparison}")

    resale_value = predict_resale_price(predicted_price, years=years_to_predict)
    st.warning(f"Estimated Resale Price after {years_to_predict} years: ₹{resale_value}")
# app.py

import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load('winning_model.pkl')
features = joblib.load('features.pkl')

# Load dataset
df = pd.read_csv('fifa_players.csv')

st.title("FIFA Player Winning Rate Predictor")

# Player selector
player_name = st.selectbox("Select a Player", df['name'].unique())

# Extract player data
player_row = df[df['name'] == player_name].iloc[0]
st.subheader("Player Information")
st.write(player_row[['full_name', 'age', 'nationality', 'positions', 'overall_rating', 'potential']])

# Prediction
input_data = player_row[features].values.reshape(1, -1)
prediction = model.predict(input_data)[0]

st.subheader("Predicted Winning Rate")
st.metric(label="Winning Probability", value=f"{prediction*100:.2f}%")

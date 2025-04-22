#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import pickle

# Load the saved model
with open('hotel_booking_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Hotel Booking Cancellation Prediction")

st.sidebar.header("Input Booking Information")

def user_input():
    data = {
        "lead_time": st.sidebar.number_input("Lead Time", 0, 500, 100),
        "avg_price_per_room": st.sidebar.number_input("Avg Price Per Room", 0.0, 10000.0, 100.0),
        "total_nights": st.sidebar.slider("Total Nights", 1, 30, 2),
        "total_guests": st.sidebar.slider("Total Guests", 1, 10, 2),
        "no_of_previous_cancellations": st.sidebar.number_input("Previous Cancellations", 0, 10, 0),
        "no_of_previous_bookings_not_canceled": st.sidebar.number_input("Prev Bookings Not Canceled", 0, 10, 0),
        "no_of_special_requests": st.sidebar.slider("Special Requests", 0, 5, 0),
        "type_of_meal_plan": st.sidebar.selectbox("Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected']),
        "room_type_reserved": st.sidebar.selectbox("Room Type", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3']),
        "market_segment_type": st.sidebar.selectbox("Market Segment", ['Online', 'Offline', 'Corporate']),
        "season": st.sidebar.selectbox("Season", ['Spring', 'Summer', 'Fall', 'Winter']),
        "arrival_month": st.sidebar.slider("Arrival Month", 1, 12, 6)
    }
    return pd.DataFrame([data])

input_df = user_input()

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    st.write(f"### Booking Status Prediction: {'Canceled' if prediction == 1 else 'Not Canceled'}")
    st.write(f"### Probability of Cancellation: {probability:.2%}")

# Test Case 1
if st.button("Test Case 1"):
    test_case_1 = {
        "lead_time": 50, "avg_price_per_room": 120.0, "total_nights": 3, "total_guests": 2,
        "no_of_previous_cancellations": 0, "no_of_previous_bookings_not_canceled": 1, "no_of_special_requests": 1,
        "type_of_meal_plan": "Meal Plan 1", "room_type_reserved": "Room_Type 1",
        "market_segment_type": "Online", "season": "Summer", "arrival_month": 7
    }
    df_test = pd.DataFrame([test_case_1])
    result = model.predict(df_test)[0]
    st.success(f"Test Case 1 Prediction: {'Canceled' if result == 1 else 'Not Canceled'}")

# Test Case 2
if st.button("Test Case 2"):
    test_case_2 = {
        "lead_time": 200, "avg_price_per_room": 300.0, "total_nights": 5, "total_guests": 4,
        "no_of_previous_cancellations": 2, "no_of_previous_bookings_not_canceled": 0, "no_of_special_requests": 3,
        "type_of_meal_plan": "Meal Plan 2", "room_type_reserved": "Room_Type 2",
        "market_segment_type": "Corporate", "season": "Winter", "arrival_month": 12
    }
    df_test = pd.DataFrame([test_case_2])
    result = model.predict(df_test)[0]
    st.success(f"Test Case 2 Prediction: {'Canceled' if result == 1 else 'Not Canceled'}")


# In[ ]:





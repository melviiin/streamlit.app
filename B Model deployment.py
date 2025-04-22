#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# In[9]:


# Load the dataset
df = pd.read_csv(r'C:\Users\Lenovo\OneDrive\Documents\Data science\smst 4\Model Deployment\2702324556\Dataset_B_hotel.csv')

df = df.copy()

# Periksa nama-nama kolom yang ada dalam dataframe
print("Kolom dalam dataset:", df.columns)

# Preprocessing data
# Gantilah nilai yang hilang pada kolom tertentu
df['type_of_meal_plan'] = df['type_of_meal_plan'].fillna('Not Selected')
df['required_car_parking_space'] = df['required_car_parking_space'].fillna(0)
df['avg_price_per_room'] = df['avg_price_per_room'].fillna(df['avg_price_per_room'].median())

# Membuat kolom baru berdasarkan data yang ada
df['total_nights'] = df['no_of_weekend_nights'] + df['no_of_week_nights']
df['total_guests'] = df['no_of_adults'] + df['no_of_children']

# Menambahkan kolom 'season' berdasarkan bulan kedatangan
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

df['season'] = df['arrival_month'].apply(get_season)

# Menghapus kolom yang tidak diperlukan
df.drop(['Booking_ID', 'arrival_date', 'arrival_year'], axis=1, inplace=True)

# Mengubah kolom-kolom kategorikal menjadi numerik dengan LabelEncoder
categorical_columns = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type', 'season']

le = LabelEncoder()

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Memisahkan fitur (X) dan target (y)
X = df.drop('booking_status', axis=1)  # 'booking_status' adalah target
y = df['booking_status'].apply(lambda x: 1 if x == 'Canceled' else 0)  # Mengubah 'Canceled' menjadi 1 dan selain itu menjadi 0

# Membagi dataset menjadi data latih dan data uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Inisialisasi model XGBoost
model = xgb.XGBClassifier(eval_metric='logloss')

# Latih model
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Tampilkan akurasi
print(f'Akurasi Model: {accuracy * 100:.2f}%')

# Simpan model yang sudah dilatih ke dalam file .pkl
with open('hotel_cancellation_xgb.pkl', 'wb') as f:
    pickle.dump(model, f)


# In[10]:


class BookingCancellationModel:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.pipeline = None

        self.numeric_features = ['lead_time', 'avg_price_per_room', 'total_nights', 'total_guests',
                                 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
                                 'no_of_special_requests']

        self.categorical_features = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type',
                                     'season', 'arrival_month']

        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), self.numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
        ])

    def train(self, X_train, y_train):
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42)
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(eval_metric='logloss', random_state=42)
        else:
            raise ValueError("Model type not recognized.")

        self.pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', self.model)
        ])

        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.pipeline.predict(X_test)
        y_prob = self.pipeline.predict_proba(X_test)[:, 1]

        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

    def save_model(self, path='hotel_booking_model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.pipeline, f)


# In[10]:


rf_model = BookingCancellationModel(model_type='random_forest')
rf_model.train(X_train, y_train)
rf_metrics = rf_model.evaluate(X_test, y_test)

xgb_model = BookingCancellationModel(model_type='xgboost')
xgb_model.train(X_train, y_train)
xgb_metrics = xgb_model.evaluate(X_test, y_test)

# Bandingkan dan Pilih Model Terbaik
print("Random Forest Metrics:", rf_metrics)
print("XGBoost Metrics:", xgb_metrics)

best_model = rf_model if rf_metrics['f1'] > xgb_metrics['f1'] else xgb_model
print("Best model:", best_model.model_type)
best_model.save_model('hotel_booking_model.pkl')


# In[11]:


import streamlit as st
import pandas as pd
import pickle

# Load model
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





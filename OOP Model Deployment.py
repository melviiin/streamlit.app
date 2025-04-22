#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
            self.model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
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

# Fungsi untuk membandingkan model dan memilih yang terbaik
def compare_models(X_train, y_train, X_test, y_test):
    rf_model = BookingCancellationModel(model_type='random_forest')
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)

    xgb_model = BookingCancellationModel(model_type='xgboost')
    xgb_model.train(X_train, y_train)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)

    print("Random Forest Metrics:", rf_metrics)
    print("XGBoost Metrics:", xgb_metrics)

    best_model = rf_model if rf_metrics['f1'] > xgb_metrics['f1'] else xgb_model
    best_model.save_model('hotel_booking_model.pkl')
    return best_model


# In[ ]:





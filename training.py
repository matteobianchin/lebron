# nba-analytics-app/training.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Carica il dataset
DATA_PATH = 'data/raw_data.csv'
data = pd.read_csv(DATA_PATH)

# Definiamo le feature e il target
features = ['days_rest', 'home_game', 'opponent_avg_points', 'travel_km', 'recent_performance']
target = 'points'

X = data[features]
y = data[target]

# Suddividiamo il dataset in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inizializziamo e alleniamo Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Salviamo il modello Random Forest
joblib.dump(rf_model, 'models/random_forest.pkl')

# Inizializziamo e alleniamo XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
xgb_model.fit(X_train, y_train)

# Salviamo il modello XGBoost
joblib.dump(xgb_model, 'models/xgboost.pkl')

print("🎉 Modelli addestrati e salvati con successo!")

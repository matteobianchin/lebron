# nba-analytics-app/utils.py

import pandas as pd
import joblib

# Caricamento dati da CSV
def load_data(path='data/raw_data.csv'):
    return pd.read_csv(path)

# Caricamento modello salvato
def load_model(path):
    return joblib.load(path)

# Preprocessing dati in ingresso per la predizione
def preprocess_input(data):
    # Prendiamo l'ultima riga (simuliamo "prossima partita")
    input_data = data.iloc[[-1]]

    # Selezioniamo solo le colonne utili per la predizione
    features = ['days_rest', 'home_game', 'opponent_avg_points', 'travel_km', 'recent_performance']

    # Verifica che le feature siano presenti nei dati
    input_data = input_data[features]

    return input_data

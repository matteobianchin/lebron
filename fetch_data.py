# nba-analytics-app/fetch_data.py

from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import numpy as np

# Trova ID giocatore LeBron James
player_dict = players.find_players_by_full_name("LeBron James")[0]
player_id = player_dict['id']

# Scarica log delle partite per la stagione 2023-24
gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2023-24', season_type_all_star='Regular Season')
data = gamelog.get_data_frames()[0]

# Pulizia dati
# Rinomina colonne per coerenza con il progetto
data = data.rename(columns={
    'GAME_DATE': 'game_date',
    'MATCHUP': 'opponent',
    'PTS': 'points',
    'REB': 'rebounds',
    'AST': 'assists',
    'WL': 'win_loss'
})

# Estrai solo il nome della squadra avversaria, mantieni casa/trasferta
data['home_game'] = data['opponent'].apply(lambda x: 1 if 'vs.' in x else 0)
data['opponent'] = data['opponent'].str.extract(r'[@vs.]\s+(\w+)$')[0]

# Aggiungi colonne simulate per features extra (da migliorare più avanti con dati veri)
data['days_rest'] = 2  # Placeholder, si può calcolare reale con calendario
data['opponent_avg_points'] = np.random.uniform(100, 120, size=len(data))
data['travel_km'] = np.random.uniform(0, 3000, size=len(data))
data['recent_performance'] = np.random.uniform(20, 35, size=len(data))

# Salva il file CSV
output_path = 'data/raw_data.csv'
data.to_csv(output_path, index=False)

print(f"✅ Dati di LeBron James salvati in {output_path}")

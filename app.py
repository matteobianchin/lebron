# -*- coding: utf-8 -*-
"""app.py

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VHeSEsF21Z4NqJ5KRej3P4bm9E95RaUV
"""

# Lebron Stats App — Versione avanzata multi-giocatore, multi-fonte, ML avanzato e heatmap completata ✅

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, teamdashboardbygeneralsplits, shotchartdetail
from requests.exceptions import RequestException
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# ================================
# 📊 Configurazioni iniziali
# ================================

st.set_page_config(page_title="NBA Player Dashboard Pro", layout="wide")
st.title("🏀 NBA Player Analytics Dashboard — Advanced Edition")

# Lista ufficiale abbreviazioni squadre NBA
TEAM_ABBREVIATIONS = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN', 'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

# Lista giocatori con ID NBA e archetipo
PLAYERS = {
    'LeBron James': {'id': 2544, 'archetype': 'Playmaking Forward'},
    'Kevin Durant': {'id': 201142, 'archetype': 'Shot Creator'},
    'Giannis Antetokounmpo': {'id': 203507, 'archetype': 'Slasher'},
    'Kawhi Leonard': {'id': 202695, 'archetype': 'Two-Way Wing'},
    'Stephen Curry': {'id': 201939, 'archetype': 'Sharpshooter'},
    'Nikola Jokic': {'id': 203999, 'archetype': 'Playmaking Big'},
    'Luka Doncic': {'id': 1629029, 'archetype': 'Playmaking Guard'}
}

# ================================
# 📅 Selezioni utente dinamiche
# ================================

seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(2003, datetime.now().year + 1)]
selected_seasons = st.multiselect("Seleziona le stagioni:", seasons, default=seasons[-3:])
player_name = st.selectbox("Seleziona il giocatore:", list(PLAYERS.keys()))
te...
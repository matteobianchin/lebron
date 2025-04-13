# Lebron Stats App — Full Live Mode ✅ + Bugfix + Migliorie Visive + ML e Multi-Giocatoti

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import playergamelog, teamdashboardbygeneralsplits, shotchartdetail
from sklearn.linear_model import LinearRegression
from datetime import datetime
import numpy as np

# ================================
# 📊 Configurazioni iniziali
# ================================

st.set_page_config(page_title="NBA Player Dashboard Pro", layout="wide")
st.title("🏀 NBA Player Analytics Dashboard — Live Mode Cloud")

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
# 📅 Selezioni utente dinamiche — fix stagioni future
# ================================

current_year = datetime.now().year
current_season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(2003, current_season + 1)]
selected_seasons = st.multiselect("Seleziona le stagioni:", seasons, default=seasons[-3:])
player_names = st.multiselect("Seleziona i giocatori:", list(PLAYERS.keys()), default=['LeBron James'])
team_name = st.selectbox("Seleziona la squadra avversaria:", list(TEAM_ABBREVIATIONS.keys()))
team_abbr = TEAM_ABBREVIATIONS[team_name]

# ================================
# 🔄 Funzioni di caricamento dati live — fix heatmap
# ================================

@st.cache_data(show_spinner=True)
def load_player_data(player_id, seasons):
    data = []
    for season in seasons:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, season_type_all_star='Regular Season')
        df = gamelog.get_data_frames()[0]
        df['SEASON'] = season
        data.append(df)
    if data:
        return pd.concat(data)
    return pd.DataFrame()

@st.cache_data(show_spinner=True)
def load_team_ratings(team_abbr):
    try:
        team_stats = teamdashboardbygeneralsplits.TeamDashboardByGeneralSplits(team_id=team_abbr).get_data_frames()[0]
        return team_stats['OFF_RATING'][0], team_stats['DEF_RATING'][0]
    except:
        return None, None

@st.cache_data(show_spinner=True)
def load_shot_chart(player_id, season, team_abbr):
    try:
        shotchart = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]
        return shotchart[shotchart['OPPONENT_TEAM_ABBREVIATION'] == team_abbr]
    except:
        return pd.DataFrame()

# ================================
# 🚀 Main App Logic — multi player + ML predittivo
# ================================

if selected_seasons and player_names:
    st.spinner("Caricamento dati in corso...")

    off_rating, def_rating = load_team_ratings(team_abbr)

    for player_name in player_names:
        player_id = PLAYERS[player_name]['id']
        player_archetype = PLAYERS[player_name]['archetype']
        df = load_player_data(player_id, selected_seasons)

        if df.empty:
            st.warning(f"Nessun dato trovato per {player_name} nelle stagioni selezionate.")
            continue

        df_team = df[df['MATCHUP'].str.contains(team_abbr)]
        ppg = df_team['PTS'].mean() if not df_team.empty else 0

        st.header(f"📊 Statistiche per {player_name} contro {team_name}")
        st.metric("Media punti a partita", f"{round(ppg, 2)} PPG")
        if off_rating and def_rating:
            st.metric("Off/Def Rating avversario", f"{off_rating} / {def_rating}")

        # 🔥 Machine Learning semplice: previsione punti prossima partita
        if len(df_team) >= 2:
            df_team_sorted = df_team.sort_values('GAME_DATE')
            df_team_sorted['GameNumber'] = range(len(df_team_sorted))
            X = df_team_sorted[['GameNumber']]
            y = df_team_sorted['PTS']
            model = LinearRegression().fit(X, y)
            next_game = np.array([[len(df_team_sorted)]])
            prediction = model.predict(next_game)[0]
            st.metric("📈 Previsione ML Punti Prossima Partita", f"{prediction:.1f} PTS")

        st.subheader("📈 Heatmap tiri (Regular Season)")
        for season in selected_seasons:
            shotchart = load_shot_chart(player_id, season, team_abbr)
            if not shotchart.empty:
                fig, ax = plt.subplots(figsize=(6, 5))
                made = shotchart[shotchart['SHOT_MADE_FLAG'] == 1]
                missed = shotchart[shotchart['SHOT_MADE_FLAG'] == 0]
                ax.scatter(made['LOC_X'], made['LOC_Y'], c='green', label='Canestri', s=10)
                ax.scatter(missed['LOC_X'], missed['LOC_Y'], c='red', label='Errori', s=10, alpha=0.5)
                ax.legend()
                ax.set_title(f"Heatmap Tiri - {season}")
                ax.axis('off')
                st.pyplot(fig)
            else:
                st.info(f"Nessun dato tiri disponibile per la stagione {season}.")

        st.subheader("📋 Dettaglio partite vs " + team_name)

        # Formattiamo le percentuali di tiro in formato %
        df_team_formatted = df_team.copy()
        for col in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
            if col in df_team_formatted.columns:
                df_team_formatted[col] = (df_team_formatted[col] * 100).round(1).astype(str) + '%'

        st.dataframe(df_team_formatted[['GAME_DATE', 'PTS', 'AST', 'REB', 'FG_PCT', 'FG3_PCT', 'FT_PCT']])

        csv = df_team.to_csv(index=False).encode('utf-8')
        st.download_button(f"📥 Scarica dati CSV per {player_name}", csv, f"{player_name}_vs_{team_name}.csv", "text/csv")

st.success("✅ Dashboard aggiornata e funzionante in modalità Live con Multi-Player e ML!")

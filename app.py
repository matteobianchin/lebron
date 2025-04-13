# Lebron Stats App — Full Live Mode ✅ + Multi-Metrica Confronto Giocatori + Trend + ML Avanzato + Multi Fonte Ready 🚀

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

PLAYERS = {
    'LeBron James': {'id': 2544, 'archetype': 'Playmaking Forward'},
    'Kevin Durant': {'id': 201142, 'archetype': 'Shot Creator'},
    'Giannis Antetokounmpo': {'id': 203507, 'archetype': 'Slasher'},
    'Kawhi Leonard': {'id': 202695, 'archetype': 'Two-Way Wing'},
    'Stephen Curry': {'id': 201939, 'archetype': 'Sharpshooter'},
    'Nikola Jokic': {'id': 203999, 'archetype': 'Playmaking Big'},
    'Luka Doncic': {'id': 1629029, 'archetype': 'Playmaking Guard'}
}

current_year = datetime.now().year
current_season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1
seasons = [f"{year}-{str(year + 1)[-2:]}" for year in range(2003, current_season + 1)]
player_names = st.multiselect("Seleziona i giocatori:", list(PLAYERS.keys()), default=['LeBron James'])
teams = list(TEAM_ABBREVIATIONS.keys())
team_name = st.selectbox("Seleziona la squadra avversaria:", teams)
team_abbr = TEAM_ABBREVIATIONS[team_name]

# ================================
# 🔄 Funzioni caricamento dati
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

# 📦 Future Integration Placeholder
# def load_external_stats():
#     # Qui integreremo future API come StatMuse o ESPN
#     return external_data

# ================================
# 🚀 Main App Logic — avanzato
# ================================

if player_names:
    off_rating, def_rating = load_team_ratings(team_abbr)

    st.header(f"📊 Confronto Giocatori — Statistiche vs {team_name}")

    comparison_data = {metric: {} for metric in ['PPG', 'AST', 'REB', 'FG%', '3P%', 'FT%']}

    tabs = st.tabs(player_names)

    for tab, player_name in zip(tabs, player_names):
        with tab:
            player_id = PLAYERS[player_name]['id']
            selected_seasons = st.multiselect(f"Seleziona stagioni per {player_name}:", seasons, default=seasons[-3:])
            df = load_player_data(player_id, selected_seasons)

            if df.empty:
                st.warning(f"Nessun dato trovato per {player_name} nelle stagioni selezionate.")
                continue

            df_team = df[df['MATCHUP'].str.contains(team_abbr)]

            if not df_team.empty:
                comparison_data['PPG'][player_name] = df_team['PTS'].mean()
                comparison_data['AST'][player_name] = df_team['AST'].mean()
                comparison_data['REB'][player_name] = df_team['REB'].mean()
                comparison_data['FG%'][player_name] = df_team['FG_PCT'].mean() * 100
                comparison_data['3P%'][player_name] = df_team['FG3_PCT'].mean() * 100
                comparison_data['FT%'][player_name] = df_team['FT_PCT'].mean() * 100

            st.subheader(f"Dati per {player_name}")
            st.metric("Media punti a partita", f"{comparison_data['PPG'].get(player_name, 0):.2f} PPG")
            if off_rating and def_rating:
                st.metric("Off/Def Rating avversario", f"{off_rating} / {def_rating}")

            # 📈 Trend stagionale punti
            if not df_team.empty:
                st.subheader("📈 Trend punti stagione")
                df_team_sorted = df_team.sort_values('GAME_DATE')
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(df_team_sorted['GAME_DATE'], df_team_sorted['PTS'], marker='o', linestyle='-')
                ax.set_xlabel("Data Partita")
                ax.set_ylabel("Punti")
                ax.set_title(f"Trend Punti - {player_name} vs {team_name}")
                plt.xticks(rotation=45)
                st.pyplot(fig)

            # 🧠 ML Avanzato: multi-feature
            if len(df_team) >= 2:
                st.subheader("🤖 Previsione ML punti prossima partita (avanzata)")
                df_team_sorted['GameNumber'] = range(len(df_team_sorted))
                features = ['GameNumber', 'AST', 'REB', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
                X = df_team_sorted[features].fillna(0)
                y = df_team_sorted['PTS']
                model = LinearRegression().fit(X, y)
                next_game_features = np.array([[len(df_team_sorted),
                                                df_team_sorted['AST'].mean(),
                                                df_team_sorted['REB'].mean(),
                                                df_team_sorted['FG_PCT'].mean(),
                                                df_team_sorted['FG3_PCT'].mean(),
                                                df_team_sorted['FT_PCT'].mean()]])
                prediction = model.predict(next_game_features)[0]
                st.metric("📈 Previsione avanzata PTS", f"{prediction:.1f} PTS")

            # 📊 Heatmap Tiri
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

            # 📋 Dettaglio partite
            st.subheader("📋 Dettaglio partite vs " + team_name)
            df_team_formatted = df_team.copy()
            for col in ['FG_PCT', 'FG3_PCT', 'FT_PCT']:
                if col in df_team_formatted.columns:
                    df_team_formatted[col] = (df_team_formatted[col] * 100).round(1).astype(str) + '%'
            st.dataframe(df_team_formatted[['GAME_DATE', 'PTS', 'AST', 'REB', 'FG_PCT', 'FG3_PCT', 'FT_PCT']])

            csv = df_team.to_csv(index=False).encode('utf-8')
            st.download_button(f"📥 Scarica dati CSV per {player_name}", csv, f"{player_name}_vs_{team_name}.csv", "text/csv")

    # 📊 Confronto multiplo migliorato per ogni metrica
    for metric, data in comparison_data.items():
        if data:
            st.subheader(f"📊 Confronto {metric} Giocatori")
            fig, ax = plt.subplots(figsize=(8, 4))
            players = list(data.keys())
            values = list(data.values())
            ax.bar(players, values, color='skyblue')
            ax.set_ylabel(metric)
            ax.set_title(f"Confronto {metric} vs {team_name}")
            st.pyplot(fig)

st.success("✅ Dashboard aggiornata: trend, ML avanzato, confronto completo e multi-fonte pronto!")

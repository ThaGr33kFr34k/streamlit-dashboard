import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import random
import ast
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Fantasy Basketball Analytics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better mobile experience
st.markdown("""
<style>
/* Überschriften */
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    color: #FF6B35;
}

/* Metrik-Container (passt sich an) */
div.st-emotion-cache-1f810n4 {
    background-color: var(--secondary-background-color) !important;
}

/* Allgemeine Styles für alle Boxen (Rahmen, Padding) */
.favorite-opponent, .nightmare-opponent, .champion-player, .legend-player, .loyalty-player {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

/* Farbanpassungen für Dark/Light-Mode mit festen Randfarben */
.favorite-opponent {
    border-left: 4px solid #4CAF50 !important;
}
.nightmare-opponent {
    border-left: 4px solid #f44336 !important;
}
.champion-player {
    border-left: 4px solid #FFD700 !important;
}
.legend-player {
    border-left: 4px solid #4169E1 !important;
}
.loyalty-player {
    border-left: 4px solid #8A2BE2 !important;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Re-add caching
def load_data():
    try:
        teams_url = "https://docs.google.com/spreadsheets/d/1xREpOPu-_5QTUzxX9I6mdqdO8xmI3Yz-uBjRBCRnyuQ/export?format=csv&gid=648434164"
        matchups_url = "https://docs.google.com/spreadsheets/d/1xREpOPu-_5QTUzxX9I6mdqdO8xmI3Yz-uBjRBCRnyuQ/export?format=csv&gid=652199133"
        drafts_url = "https://docs.google.com/spreadsheets/d/1xREpOPu-_5QTUzxX9I6mdqdO8xmI3Yz-uBjRBCRnyuQ/export?format=csv&gid=2084485780"
        categories_url = "https://docs.google.com/spreadsheets/d/1xREpOPu-_5QTUzxX9I6mdqdO8xmI3Yz-uBjRBCRnyuQ/export?format=csv&gid=987718515"
        seasons_url = "https://docs.google.com/spreadsheets/d/1xREpOPu-_5QTUzxX9I6mdqdO8xmI3Yz-uBjRBCRnyuQ/export?format=csv&gid=1895764019"
        trades_url = "https://docs.google.com/spreadsheets/d/1xREpOPu-_5QTUzxX9I6mdqdO8xmI3Yz-uBjRBCRnyuQ/export?format=csv&gid=58770562"
        
        teams_df = pd.read_csv(teams_url)
        matchups_df = pd.read_csv(matchups_url)
        drafts_df = pd.read_csv(drafts_url)
        categories_df = pd.read_csv(categories_url)
        seasons_df = pd.read_csv(seasons_url)
        trades_df = pd.read_csv(trades_url)
        
        # --- HIER WIRD DIE SPALTE UMBENANNT ---
        if 'Year' in seasons_df.columns:
            seasons_df = seasons_df.rename(columns={'Year': 'Saison'})
            
        # Optional: Clean up seasons_df columns
        seasons_df.columns = seasons_df.columns.str.strip()
        
        return teams_df, matchups_df, drafts_df, categories_df, seasons_df, trades_df
    
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        return None, None, None, None, None, None
        
def create_team_mapping(teams_df):
    """Create mapping from TeamID to Manager Name by year"""
    if teams_df is None:
        return {}
    
    # Create a mapping dictionary: {(TeamID, Year): FirstName}
    mapping = {}
    for _, row in teams_df.iterrows():
        key = (row['TeamID'], row['Year'])
        mapping[key] = row['First Name']
    
    return mapping

def parse_pick_order(pick_order_str):
    """Parse the pickOrder string from [10, 2, 3, ...] format"""
    try:
        return ast.literal_eval(pick_order_str)
    except:
        return None

def calculate_correlation(x, y):
    """Calculate Pearson correlation coefficient manually"""
    if len(x) != len(y) or len(x) == 0:
        return 0, 1
    
    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Calculate means
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Calculate correlation coefficient
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    
    if denominator == 0:
        return 0, 1
    
    correlation = numerator / denominator
    
    # Simple p-value approximation (not exact but good enough for our use case)
    n = len(x)
    if n > 2:
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2 + 1e-8))
        # Rough p-value approximation
        p_value = 2 * (1 - np.minimum(0.999, np.abs(t_stat) / np.sqrt(n)))
    else:
        p_value = 1
    
    return correlation, p_value

def process_matchup_data(matchups_df, team_mapping):
    """Process matchup data and add manager names"""
    if matchups_df is None or not team_mapping:
        return None
    
    processed_matches = []
    
    for _, match in matchups_df.iterrows():
        try:
            # Get manager names for home and away teams
            home_manager = team_mapping.get((match['Home'], match['Season']), f"Team_{match['Home']}")
            away_manager = team_mapping.get((match['Away'], match['Season']), f"Team_{match['Away']}")
            
            # Add processed match
            processed_match = match.copy()
            processed_match['Home_Manager'] = home_manager
            processed_match['Away_Manager'] = away_manager
            processed_matches.append(processed_match)
            
        except Exception as e:
            continue
    
    return pd.DataFrame(processed_matches)

def process_draft_data(drafts_df, teams_df):
    """Process draft data and combine with final rankings"""
    if drafts_df is None or teams_df is None:
        return None
    
    # Get pick order for each season
    pick_orders = {}
    
    # Extract pick order from drafts_df (assuming it's in a column called 'pickOrder')
    if 'pickOrder' in drafts_df.columns:
        for season in drafts_df['Season'].unique():
            season_data = drafts_df[drafts_df['Season'] == season]
            if not season_data.empty and 'pickOrder' in season_data.columns:
                pick_order_str = season_data['pickOrder'].iloc[0]
                if pd.notna(pick_order_str):
                    pick_orders[season] = parse_pick_order(pick_order_str)
    
    # Create draft analysis data
    draft_analysis = []
    
    for season in teams_df['Year'].unique():
        season_teams = teams_df[teams_df['Year'] == season]
        
        if season in pick_orders:
            pick_order = pick_orders[season]
            
            for i, team_id in enumerate(pick_order):
                draft_position = i + 1  # 1-indexed draft position
                
                # Find this team's final rank
                team_info = season_teams[season_teams['TeamID'] == team_id]
                if not team_info.empty:
                    final_rank = team_info['Final Rank'].iloc[0]
                    manager_name = team_info['First Name'].iloc[0]
                    
                    # Calculate over/under (positive = overperformed, negative = underperformed)
                    over_under = draft_position - final_rank
                    
                    draft_analysis.append({
                        'Season': season,
                        'TeamID': team_id,
                        'Manager': manager_name,
                        'Draft_Position': draft_position,
                        'Final_Rank': final_rank,
                        'Over_Under': over_under
                    })
    
    return pd.DataFrame(draft_analysis)

def calculate_draft_value_analysis(draft_analysis_df):
    """Calculate average final rank by draft position"""
    if draft_analysis_df is None:
        return None
    
    # Group by draft position and calculate average final rank
    draft_value = draft_analysis_df.groupby('Draft_Position')['Final_Rank'].agg([
        'mean', 'std', 'count'
    ]).round(2)
    
    draft_value.columns = ['Avg_Final_Rank', 'Std_Final_Rank', 'Count']
    draft_value = draft_value.reset_index()
    
    return draft_value

def create_draft_scatter_plot(draft_analysis_df):
    """Create scatter plot showing draft position vs final rank"""
    if draft_analysis_df is None:
        return None
    
    # Create scatter plot
    fig = px.scatter(
        draft_analysis_df, 
        x='Draft_Position', 
        y='Final_Rank',
        color='Over_Under',
        size='Season',
        hover_data=['Manager', 'Season'],
        title='Draft Position vs Final Rank (alle Saisons)',
        labels={
            'Draft_Position': 'Draft Position', 
            'Final_Rank': 'Final Rank',
            'Over_Under': 'Over/Under Score',
        },
        template="plotly_dark",
        color_continuous_scale='RdYlGn'  # Red for negative, Green for positive
    )
    
    # Add diagonal line (perfect prediction)
    max_pos = max(draft_analysis_df['Draft_Position'].max(), draft_analysis_df['Final_Rank'].max())
    fig.add_trace(go.Scatter(
        x=[1, max_pos],
        y=[1, max_pos],
        mode='lines',
        name='Perfekte Vorhersage',
        line=dict(color='black', dash='dash'),
        opacity=0.5
    ))
    
    fig.update_layout(
        yaxis=dict(autorange='reversed'),  # Lower ranks at top
        height=500
    )
    
    return fig

def calculate_cumulative_over_under(draft_analysis_df):
    """Calculate cumulative over/under scores by manager across all seasons"""
    if draft_analysis_df is None:
        return None
    
    # Group by manager and sum over/under scores
    cumulative_df = draft_analysis_df.groupby('Manager').agg({
        'Over_Under': 'sum',
        'Season': 'count',  # Number of seasons
        'Draft_Position': 'mean',  # Average draft position
        'Final_Rank': 'mean'  # Average final rank
    }).round(2)
    
    cumulative_df.columns = ['Kumulierter_Over_Under', 'Anzahl_Saisons', 'Avg_Draft_Position', 'Avg_Final_Rank']
    cumulative_df = cumulative_df.reset_index()
    
    # Sort by cumulative over/under (best performers first)
    cumulative_df = cumulative_df.sort_values('Kumulierter_Over_Under', ascending=False)
    
    return cumulative_df

def calculate_head_to_head(processed_df, manager1, manager2):
    """Calculate head-to-head stats between two managers"""
    if processed_df is None:
        return None
    
    # Find all matches between these managers
    h2h_matches = processed_df[
        ((processed_df['Home_Manager'] == manager1) & (processed_df['Away_Manager'] == manager2)) |
        ((processed_df['Home_Manager'] == manager2) & (processed_df['Away_Manager'] == manager1))
    ].copy()
    
    if len(h2h_matches) == 0:
        return {"games": 0, "wins": 0, "losses": 0, "ties": 0, "win_pct": 0}
    
    # Calculate wins for manager1
    wins = 0
    losses = 0
    ties = 0
    
    for _, match in h2h_matches.iterrows():
        if match['Winner'] == 'TIE':
            ties += 1
        elif (match['Home_Manager'] == manager1 and match['Winner'] == 'HOME') or \
             (match['Away_Manager'] == manager1 and match['Winner'] == 'AWAY'):
            wins += 1
        else:
            losses += 1
    
    total_games = len(h2h_matches)
    win_pct = wins / total_games if total_games > 0 else 0
    
    return {
        "games": total_games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_pct": win_pct
    }

def calculate_all_h2h_stats(processed_df, selected_manager, min_games=5):
    """Calculate H2H stats for selected manager against all other managers"""
    if processed_df is None:
        return None, None
    
    # Get all unique managers
    all_managers = set(processed_df['Home_Manager'].unique()) | set(processed_df['Away_Manager'].unique())
    all_managers.discard(selected_manager)  # Remove selected manager from opponents
    
    h2h_results = []
    
    for opponent in all_managers:
        h2h_stats = calculate_head_to_head(processed_df, selected_manager, opponent)
        
        if h2h_stats['games'] >= min_games:  # Minimum games filter
            h2h_results.append({
                'Opponent': opponent,
                'Games': h2h_stats['games'],
                'Wins': h2h_stats['wins'],
                'Losses': h2h_stats['losses'],
                'Ties': h2h_stats['ties'],
                'Win%': h2h_stats['win_pct']
            })
    
    if not h2h_results:
        return None, None
    
    h2h_df = pd.DataFrame(h2h_results)
    
    # Sort by win percentage
    favorites = h2h_df.nlargest(3, 'Win%')  # Top 3 highest win%
    nightmares = h2h_df.nsmallest(3, 'Win%')  # Top 3 lowest win%
    
    return favorites, nightmares

def calculate_playoff_stats(processed_df, teams_df):
    """Calculate Regular vs Playoff performance per manager"""
    if processed_df is None or teams_df is None:
        return None, None, None

    # Filter out LOSERS_CONSOLATION_LADDER games from all analysis
    filtered_df = processed_df[processed_df['Phase'] != 'LOSERS_CONSOLATION_LADDER'].copy()

    # Define phases correctly - handle case-insensitive matching
    regular_phases = ["Regular Season"]
    # Include both uppercase and lowercase variants to handle inconsistent data
    playoff_phases = ["FINALE", "Finale", "finale", "Halbfinale", "WINNERS_BRACKET", "Spiel um Platz 3"]
    
    stats = []

    # Get all managers from teams data
    managers = teams_df['First Name'].unique()

    for manager in managers:
        # Get all team IDs for this manager across all years
        manager_teams = teams_df[teams_df['First Name'] == manager]
        
        # Regular Season Stats
        reg_wins = 0
        reg_total = 0
        
        # Playoff Stats  
        playoff_wins = 0
        playoff_total = 0
        
        # Go through each team/year combination for this manager
        for _, team_row in manager_teams.iterrows():
            team_id = team_row['TeamID']
            year = team_row['Year']
            
            # Regular season games for this team/year
            reg_games = filtered_df[
                (filtered_df['Phase'].isin(regular_phases)) &
                (filtered_df['Season'] == year) &
                ((filtered_df['Home'] == team_id) | (filtered_df['Away'] == team_id))
            ]
            
            reg_total += len(reg_games)
            
            # Count wins in regular season
            for _, game in reg_games.iterrows():
                if (game['Home'] == team_id and game['Winner'] == 'HOME') or \
                   (game['Away'] == team_id and game['Winner'] == 'AWAY'):
                    reg_wins += 1
            
            # Playoff games for this team/year
            playoff_games = filtered_df[
                (filtered_df['Phase'].isin(playoff_phases)) &
                (filtered_df['Season'] == year) &
                ((filtered_df['Home'] == team_id) | (filtered_df['Away'] == team_id))
            ]
            
            playoff_total += len(playoff_games)
            
            # Count wins in playoffs
            for _, game in playoff_games.iterrows():
                if (game['Home'] == team_id and game['Winner'] == 'HOME') or \
                   (game['Away'] == team_id and game['Winner'] == 'AWAY'):
                    playoff_wins += 1
        
        # Calculate win percentages
        reg_win_pct = reg_wins / reg_total if reg_total > 0 else 0.0
        playoff_win_pct = playoff_wins / playoff_total if playoff_total > 0 else 0.0

        stats.append({
            "Manager": manager,
            "Regular Games": reg_total,
            "Regular Wins": reg_wins,
            "Regular Win%": round(reg_win_pct, 3),
            "Playoff Games": playoff_total,
            "Playoff Wins": playoff_wins,
            "Playoff Win%": round(playoff_win_pct, 3),
        })

    df = pd.DataFrame(stats)
    
    # Create ranking tables
    reg_ranked = df[['Manager', 'Regular Games', 'Regular Wins', 'Regular Win%']].sort_values(
        by="Regular Win%", ascending=False
    ).reset_index(drop=True)
    
    playoff_ranked = df[['Manager', 'Playoff Games', 'Playoff Wins', 'Playoff Win%']].sort_values(
        by="Playoff Win%", ascending=False
    ).reset_index(drop=True)

    return df, reg_ranked, playoff_ranked

# UPDATED PLAYER ANALYSIS FUNCTIONS - NOW BASED ON mDrafts DATA
def process_player_draft_data(drafts_df, teams_df):
    """Process player draft data from mDrafts sheet and team information"""
    if drafts_df is None or teams_df is None:
        return None
    
    player_data = []
    
    # Check if we have PlayerID and PlayerName columns as specified
    if 'PlayerID' not in drafts_df.columns or 'PlayerName' not in drafts_df.columns:
        st.error("mDrafts sheet must have 'PlayerID' and 'PlayerName' columns")
        return None
    
    # Group by season and process each draft
    for season in drafts_df['Season'].unique():
        season_drafts = drafts_df[drafts_df['Season'] == season]
        
        for _, draft_row in season_drafts.iterrows():
            team_id = draft_row['TeamID']
            player_id = draft_row['PlayerID']
            player_name = draft_row['PlayerName']
            
            if pd.isna(player_name) or pd.isna(team_id):
                continue
                
            # Find manager and team performance for this season/team
            team_info = teams_df[(teams_df['TeamID'] == team_id) & (teams_df['Year'] == season)]
            
            if not team_info.empty:
                manager_name = team_info['First Name'].iloc[0]
                final_rank = team_info['Final Rank'].iloc[0]
                
                # Determine draft position (you may need to add this logic based on your data structure)
                # For now, we'll use row index as draft position approximation
                draft_position = len(player_data) + 1  # Simple increment
                
                player_data.append({
                    'Season': season,
                    'Player': player_name,
                    'PlayerID': player_id,
                    'Manager': manager_name,
                    'TeamID': team_id,
                    'Draft_Position': draft_position,
                    'Final_Rank': final_rank,
                    'Made_Playoffs': final_rank <= 8,  # Assuming top 8 make playoffs
                    'Won_Championship': final_rank == 1,
                    'Made_Finals': final_rank <= 2
                })
    
    return pd.DataFrame(player_data) if player_data else None

def calculate_championship_dna(drafts_df, teams_df):
    """Calculate championship DNA - which players were most often on championship teams"""
    
    # Process player data from actual draft data
    player_data = process_player_draft_data(drafts_df, teams_df)
    
    if player_data is None or player_data.empty:
        st.info("Keine Spielerdaten im mDrafts Sheet gefunden. Überprüfe bitte die Spalten PlayerID und PlayerName.")
        return None, None
    
    # Calculate championship players
    championship_players = player_data[player_data['Won_Championship'] == True]
    champ_counts = championship_players['Player'].value_counts().reset_index()
    champ_counts.columns = ['Player', 'Championships']
    
    # Add championship years
    champ_data = []
    for _, player_row in champ_counts.iterrows():  # Fixed: removed asterisks
        player_name = player_row['Player']
        championships = player_row['Championships']
        
        # Get years when this player won championships
        champ_years = championship_players[championship_players['Player'] == player_name]['Season'].tolist()
        champ_years_str = ', '.join(map(str, sorted(champ_years)))
        
        champ_data.append({
            'Player': player_name,
            'Championships': championships,
            'Championship_Years': champ_years_str
        })
    
    champ_df = pd.DataFrame(champ_data).sort_values('Championships', ascending=False) if champ_data else None
    
    # Calculate finals appearances
    finals_players = player_data[player_data['Made_Finals'] == True]
    finals_counts = finals_players['Player'].value_counts().reset_index()
    finals_counts.columns = ['Player', 'Finals_Appearances']
    
    finals_data = []
    for _, player_row in finals_counts.iterrows():  # Fixed: removed asterisks
        player_name = player_row['Player']
        finals_apps = player_row['Finals_Appearances']
        
        # Get championships for this player
        player_champs = len(championship_players[championship_players['Player'] == player_name])
        finals_win_rate = player_champs / finals_apps if finals_apps > 0 else 0
        
        finals_data.append({
            'Player': player_name,
            'Finals_Appearances': finals_apps,
            'Championships': player_champs,
            'Finals_Win_Rate': finals_win_rate
        })
    
    finals_df = pd.DataFrame(finals_data).sort_values(['Finals_Appearances', 'Championships'], ascending=False) if finals_data else None
    
    # Filter for players on teams with a final rank between 3 and 8
    contender_players = player_data[
        (player_data['Final_Rank'] >= 3) & (player_data['Final_Rank'] <= 8)
    ]
    contender_counts = contender_players['Player'].value_counts().reset_index()
    contender_counts.columns = ['Player', 'Contending_Seasons']
    
    # Build a DataFrame with details for contending players
    contender_data = []
    for _, player_row in contender_counts.iterrows():  # Fixed: removed asterisks
        player_name = player_row['Player']
        contending_seasons_count = player_row['Contending_Seasons']
        
        # Get the years this player was on a contending team
        contending_years = contender_players[contender_players['Player'] == player_name]['Season'].tolist()
        contending_years_str = ', '.join(map(str, sorted(contending_years)))
        
        # Calculate average and best playoff rank for this player
        player_ranks = contender_players[contender_players['Player'] == player_name]['Final_Rank']
        avg_rank = player_ranks.mean() if not player_ranks.empty else 0
        best_rank = player_ranks.min() if not player_ranks.empty else 0
        
        contender_data.append({
            'Player': player_name,
            'Contending_Seasons': contending_seasons_count,
            'Contending_Years': contending_years_str,  # Fixed: added missing comma
            'Avg_Playoff_Rank': round(avg_rank, 1),   # Fixed: added missing comma
            'Best_Playoff_Rank': int(best_rank)       # Fixed: now variables are defined
        })
        
    contender_df = pd.DataFrame(contender_data).sort_values('Contending_Seasons', ascending=False) if contender_data else None
    
    return champ_df, finals_df, contender_df  # Fixed: added missing contender_df to return
    

def calculate_legend_analysis(drafts_df, teams_df, contender_df):
    """Calculate legend analysis - first round superstars and playoff heroes"""

    if drafts_df is None or drafts_df.empty:
        st.info("Keine Spielerdaten für Legend Analysis verfügbar.")
        return None, None
    
    # First Round Superstars - directly from Round column
    first_round_drafts = drafts_df[drafts_df['Round'] == 1]
    
    # Count how many times each PlayerID was drafted in Round 1
    first_round_counts = first_round_drafts['PlayerID'].value_counts().reset_index()
    first_round_counts.columns = ['PlayerID', 'First_Round_Picks']
    
    # Add player names by merging with the drafts data
    first_round_data = []
    for _, row in first_round_counts.iterrows():
        player_id = row['PlayerID']
        first_round_picks = row['First_Round_Picks']
        
        # Get player name from drafts_df
        player_name = drafts_df[drafts_df['PlayerID'] == player_id]['PlayerName'].iloc[0]
        
        # Get years range when this player was drafted in Round 1
        player_seasons = first_round_drafts[first_round_drafts['PlayerID'] == player_id]['Season']
        min_year = player_seasons.min()
        max_year = player_seasons.max()
        years_range = f"{min_year}-{max_year}" if min_year != max_year else str(min_year)
        
        # Calculate average pick position within Round 1
        round1_picks = first_round_drafts[first_round_drafts['PlayerID'] == player_id]['Pick']
        avg_pick = round1_picks.mean()
        
        first_round_data.append({
            'Player': player_name,
            'PlayerID': player_id,
            'First_Round_Picks': first_round_picks,
            'Avg_Pick_in_Round1': round(avg_pick, 1),
            'Years_as_Superstar': years_range
        })
    
    first_round_df = pd.DataFrame(first_round_data).sort_values('First_Round_Picks', ascending=False) if first_round_data else None
    
    # GEÄNDERT: Playoff Heroes aus contender_df mit korrekten Spalten erstellen
    if contender_df is None or contender_df.empty:
        playoff_heroes_df = None
    else:
        playoff_heroes_df = contender_df.copy()
        
        # GEÄNDERT: Erstelle die fehlenden Spalten für die HTML-Anzeige
        playoff_heroes_df['Playoff_Hero_Seasons'] = playoff_heroes_df['Contending_Seasons']
        
        # GEÄNDERT: Berechne eine Playoff Rate basierend auf verfügbaren Daten
        # Annahme: Wenn jemand X Contending Seasons hat, schätzen wir seine Gesamtsaisons
        # und berechnen daraus eine Rate (vereinfacht)
        max_seasons = drafts_df['Season'].nunique() if 'Season' in drafts_df.columns else 10
        playoff_heroes_df['Playoff_Rate'] = playoff_heroes_df['Contending_Seasons'] / max_seasons
        
        # GEÄNDERT: Verwende Avg_Playoff_Rank als Draft Position (oder erstelle Dummy-Werte)
        playoff_heroes_df['Avg_Draft_Position'] = playoff_heroes_df['Avg_Playoff_Rank']
        
        # GEÄNDERT: Berechne einen Hidden Gem Score basierend auf verfügbaren Daten
        playoff_heroes_df['Hidden_Gem_Score'] = (
            playoff_heroes_df['Contending_Seasons'] * 3 + 
            (9 - playoff_heroes_df['Best_Playoff_Rank']) * 2
        ).round(1)
        
        # GEÄNDERT: Erstelle Playoff_Appearances aus Contending_Seasons
        playoff_heroes_df['Playoff_Appearances'] = playoff_heroes_df['Contending_Seasons']
    
    return first_round_df, playoff_heroes_df


def calculate_manager_player_loyalty(drafts_df, teams_df):
    """Calculate manager-player loyalty stats directly from mDrafts sheet"""
    
    if drafts_df is None or drafts_df.empty:
        st.info("Keine Draft-Daten verfügbar.")
        return None
    
    # Überprüfe notwendige Spalten (die tatsächlich in mDrafts vorhanden sind)
    required_cols = ['TeamID', 'PlayerID', 'PlayerName']
    missing_cols = [col for col in required_cols if col not in drafts_df.columns]
    
    if missing_cols:
        st.error(f"Fehlende Spalten in mDrafts: {missing_cols}")
        return None
    
    # Finde Season/Year Spalte
    season_col = None
    for col in ['Season', 'Year', 'season', 'year']:
        if col in drafts_df.columns:
            season_col = col
            break
    
    if season_col is None:
        st.error("Keine Season/Year Spalte in mDrafts gefunden!")
        return None
    
    # Bestimme welche Draft-Positions-Spalte verwendet werden soll
    draft_pos_col = None
    if 'Pick' in drafts_df.columns:
        draft_pos_col = 'Pick'
    elif 'Draft_Position' in drafts_df.columns:
        draft_pos_col = 'Draft_Position'
    elif 'Position' in drafts_df.columns:
        draft_pos_col = 'Position'
    
    # Bestimme welche Round-Spalte verwendet werden soll
    round_col = None
    if 'Round' in drafts_df.columns:
        round_col = 'Round'
    
    # GEÄNDERT: Vereinfachte Lösung - arbeite direkt mit verfügbaren Spalten
    try:
        # Mappe TeamID zu Manager Namen
        if teams_df is not None and not teams_df.empty:
            manager_col = None
            if 'First Name' in teams_df.columns:
                manager_col = 'First Name'
            elif 'Manager' in teams_df.columns:
                manager_col = 'Manager'
            elif 'Name' in teams_df.columns:
                manager_col = 'Name'
            
            if manager_col is None:
                st.error(f"Keine Manager-Name Spalte in teams_df gefunden. Verfügbare Spalten: {teams_df.columns.tolist()}")
                return None
            
            # GEÄNDERT: Einfacher Merge ohne Saison-spezifisches Mapping
            # Erstelle Team-Manager Mapping (nimm den ersten Manager pro TeamID)
            team_manager_map = teams_df.groupby('TeamID')[manager_col].first().to_dict()
            
            # Füge Manager-Namen zu drafts_df hinzu
            drafts_with_managers = drafts_df.copy()
            drafts_with_managers['Manager'] = drafts_with_managers['TeamID'].map(team_manager_map)
            
        else:
            # Fallback: verwende TeamID als Manager
            drafts_with_managers = drafts_df.copy()
            drafts_with_managers['Manager'] = drafts_with_managers['TeamID']
        
        # GEÄNDERT: Gruppiere nach Manager und PlayerID
        loyalty_data = []
        for (manager, player_id), group in drafts_with_managers.groupby(['Manager', 'PlayerID']):
            # Hole Spielernamen
            player_name = group['PlayerName'].iloc[0]
            
            # Zähle Drafts
            times_drafted = len(group)
            
            # Hole Jahre
            years = sorted(group[season_col].unique())
            years_str = ', '.join(map(str, years))
            
            # Berechne Durchschnitte
            avg_round = group[round_col].mean() if round_col and round_col in group.columns else None
            avg_pick = group[draft_pos_col].mean() if draft_pos_col and draft_pos_col in group.columns else None
            
            # Berechne Loyalty Score
            loyalty_score = times_drafted * 3 + len(years) * 2
            if avg_round:
                loyalty_score += (15 - avg_round) * 0.5  # Bonus für frühe Runden
            
            loyalty_data.append({
                'Manager': manager,
                'Player': player_name,
                'PlayerID': player_id,
                'Times_Drafted': times_drafted,
                'Unique_Seasons': len(years),
                'Years': years_str,
                'Avg_Draft_Round': round(avg_round, 1) if avg_round else None,
                'Avg_Draft_Position': round(avg_pick, 1) if avg_pick else None,
                'Loyalty_Score': round(loyalty_score, 1)
            })
        
        # Erstelle DataFrame
        loyalty_df = pd.DataFrame(loyalty_data)
        
        # Filter: nur Manager-Spieler mit mehr als 1 Draft
        loyalty_df = loyalty_df[loyalty_df['Times_Drafted'] > 1]
        
        # Sortiere nach Loyalty Score
        loyalty_df = loyalty_df.sort_values('Loyalty_Score', ascending=False)
        
        # Entferne NaN Werte
        loyalty_df = loyalty_df.dropna(subset=['Manager', 'Player'])
        
        return loyalty_df
        
    except Exception as e:
        st.error(f"Fehler bei Loyalty-Berechnung: {str(e)}")
        return None
    
def style_dataframe_with_colors(df, win_pct_columns):
    """Apply color formatting to dataframe based on win percentage with gradient"""
    def highlight_winpct(val):
        if pd.isna(val) or not isinstance(val, (int, float)):
            return ""
        
        # Create gradient colors based on win percentage
        if val >= 0.750:
            return "background-color: rgba(0, 150, 0, 0.4);"  # Dark green
        elif val >= 0.650:
            return "background-color: rgba(50, 180, 50, 0.3);"  # Medium-dark green
        elif val >= 0.550:
            return "background-color: rgba(100, 200, 100, 0.25);"  # Medium green
        elif val > 0.500:
            return "background-color: rgba(150, 220, 150, 0.2);"  # Light green
        elif val == 0.500:
            return "background-color: rgba(255, 255, 0, 0.1);"  # Very light yellow
        elif val >= 0.450:
            return "background-color: rgba(255, 200, 150, 0.2);"  # Light orange
        elif val >= 0.350:
            return "background-color: rgba(255, 150, 100, 0.25);"  # Medium orange
        elif val >= 0.250:
            return "background-color: rgba(255, 100, 50, 0.3);"  # Medium-dark orange/red
        else:
            return "background-color: rgba(200, 0, 0, 0.4);"  # Dark red
    
    def make_manager_bold(val):
        return "font-weight: bold;"
    
    styled_df = df.style.applymap(highlight_winpct, subset=win_pct_columns)
    styled_df = styled_df.applymap(make_manager_bold, subset=['Manager'])
    return styled_df

def create_medal_table(teams_df):
    """Create Olympic-style medal table"""
    if teams_df is None:
        return None
    
    # Count medals for each manager
    medal_counts = []
    managers = teams_df['First Name'].unique()
    
    for manager in managers:
        manager_data = teams_df[teams_df['First Name'] == manager]
        
        gold = len(manager_data[manager_data['Final Rank'] == 1])
        silver = len(manager_data[manager_data['Final Rank'] == 2]) 
        bronze = len(manager_data[manager_data['Final Rank'] == 3])
        total = gold + silver + bronze
        
        medal_counts.append({
            'Manager': manager,
            'Gold': gold,
            'Silver': silver,
            'Bronze': bronze,
            'Total': total
        })
    
    # Create DataFrame
    medal_df = pd.DataFrame(medal_counts)
    
    # Olympic sorting:
    # 1. By Gold (descending)
    # 2. By Silver (descending) 
    # 3. By Bronze (descending)
    # 4. Alphabetically by Manager Name (ascending)
    medal_df_sorted = medal_df.sort_values(
        by=['Gold', 'Silver', 'Bronze', 'Manager'],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)
    
    # Add rank (same rank for ties)
    medal_df_sorted['Rank'] = 1
    
    for i in range(1, len(medal_df_sorted)):
        current = medal_df_sorted.iloc[i]
        previous = medal_df_sorted.iloc[i-1]
        
        # Same medal distribution = same rank
        if (current['Gold'] == previous['Gold'] and 
            current['Silver'] == previous['Silver'] and 
            current['Bronze'] == previous['Bronze']):
            medal_df_sorted.iloc[i, medal_df_sorted.columns.get_loc('Rank')] = medal_df_sorted.iloc[i-1]['Rank']
        else:
            medal_df_sorted.iloc[i, medal_df_sorted.columns.get_loc('Rank')] = i + 1
    
    # Reorder columns
    medal_df_final = medal_df_sorted[['Rank', 'Manager', 'Gold', 'Silver', 'Bronze', 'Total']]
    
    return medal_df_final

def display_opponent_analysis(favorites, nightmares, selected_manager):
    """Display the favorite and nightmare opponents analysis"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 😍 Lieblingsgegner")
        st.markdown("*Höchste Siegquote gegen diese Gegner*")
        
        if favorites is not None and len(favorites) > 0:
            for i, (_, opponent) in enumerate(favorites.iterrows()):
                with st.container():
                    st.markdown(f"""
                    <div class="favorite-opponent">
                        <h4>#{i+1} {opponent['Opponent']}</h4>
                        <p><strong>{opponent['Win%']:.1%}</strong> Siegquote ({opponent['Wins']}-{opponent['Losses']}-{opponent['Ties']} in {opponent['Games']} Spielen)</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(f"Keine Lieblingsgegner mit mindestens 5 Spielen für {selected_manager} gefunden.")
    
    with col2:
        st.markdown("### 😰 Angstgegner")
        st.markdown("*Niedrigste Siegquote gegen diese Gegner*")
        
        if nightmares is not None and len(nightmares) > 0:
            for i, (_, opponent) in enumerate(nightmares.iterrows()):
                with st.container():
                    st.markdown(f"""
                    <div class="nightmare-opponent">
                        <h4>#{i+1} {opponent['Opponent']}</h4>
                        <p><strong>{opponent['Win%']:.1%}</strong> Siegquote ({opponent['Wins']}-{opponent['Losses']}-{opponent['Ties']} in {opponent['Games']} Spielen)</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info(f"Keine Angstgegner mit mindestens 5 Spielen für {selected_manager} gefunden.")

def normalize_year_column(df):
    """Standardisiert Jahr-Spalten zu 'Year'"""
    if df is None or df.empty:
        return df
    
    year_columns = ['Year', 'year', 'YEAR', 'Season', 'season', 'SEASON', 
                   'Saison', 'saison', 'SAISON', 'Date', 'date', 'DATE']
    
    for col in year_columns:
        if col in df.columns:
            if col != 'Year':
                df = df.rename(columns={col: 'Year'})
            return df
    
    # Falls keine Jahr-Spalte gefunden wird, gib DataFrame unverändert zurück
    return df

def normalize_dataframes(*dataframes):
    """Normalisiert mehrere DataFrames auf einmal"""
    normalized = []
    for df in dataframes:
        normalized.append(normalize_year_column(df))
    return normalized

def _display_season_draft(manager_drafts, year, year_col):
    """Hilfsfunktion zur Anzeige einer einzelnen Saison-Draft-Tabelle (mobile-optimiert)"""
    if year_col:
        year_drafts = manager_drafts[manager_drafts[year_col] == year]
        year_display = str(year)
    else:
        year_drafts = manager_drafts
        year_display = "Alle Jahre"
        
    if not year_drafts.empty:
        # Sortiere nach Draft-Position/Pick
        pick_col = None
        for col in ['Pick', 'Draft_Pick', 'Position', 'Draft_Position']:
            if col in year_drafts.columns:
                pick_col = col
                break
        
        # Sortiere nach verfügbarer Spalte
        if pick_col:
            year_drafts = year_drafts.sort_values(pick_col, ascending=True)
        
        # Container für jede Saison (kompakt)
        with st.expander(f"**{year_display}** ({len(year_drafts)} Picks)", expanded=False):
            
            # GEÄNDERT: Verwende st.dataframe für bessere Mobile-Darstellung
            # Erstelle DataFrame für die Anzeige
            display_data = []
            
            for i, (_, pick) in enumerate(year_drafts.iterrows()):
                player_name = pick.get('PlayerName', pick.get('Spieler', 'Unbekannt'))
                position = pick.get('Pick', pick.get('Pos', 'N/A'))
                
                # Top 3 Picks markieren
                pick_display = f"#{i+1}"
                if i < 3:
                    # Füge Emoji für Top 3 hinzu
                    if i == 0:
                        pick_display = f"🥇 #{i+1}"
                    elif i == 1:
                        pick_display = f"🥈 #{i+1}"
                    elif i == 2:
                        pick_display = f"🥉 #{i+1}"
                
                display_data.append({
                    "Round": pick_display,
                    "Spieler": player_name,
                    "Pick": str(position)
                })
            
            # GEÄNDERT: Erstelle DataFrame und zeige als Tabelle
            df_display = pd.DataFrame(display_data)
            
            # GEÄNDERT: Verwende st.dataframe mit angepassten Einstellungen
            st.dataframe(
                df_display,
                use_container_width=True,  # Nutzt volle Breite
                hide_index=True,  # Versteckt Index
                height=min(400, len(df_display) * 35 + 50)  # Dynamische Höhe
            )
                    
# Main app
def main():
    global draft_analysis_df
    st.markdown('<h1 class="main-header">🏀 Fantasy Basketball Analytics</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        teams_df, matchups_df, drafts_df, categories_df, seasons_df, trades_df = load_data()
    
    if teams_df is None or matchups_df is None:
        st.error("Please update the Google Sheets URLs in the code with your actual sheet URLs.")
        st.info("""
        To get your Google Sheets CSV URLs:
        1. Open your Google Sheet
        2. Go to File → Share → Publish to web
        3. Select the tab (mTeams, mMatchups, or mDrafts)
        4. Choose CSV format
        5. Copy the generated URL
        6. Replace the URLs in the code
        """)
        return
    
    # Process data
    team_mapping = create_team_mapping(teams_df)
    processed_df = process_matchup_data(matchups_df, team_mapping)

    # Draft Analysis für alle Bereiche verfügbar machen
    if drafts_df is not None:
            draft_analysis_df = process_draft_data(drafts_df, teams_df)
    else:
        draft_analysis_df = None    
    
    if processed_df is None:
        st.error("Error processing matchup data.")
        return
    
    # Sidebar with button navigation
    st.sidebar.title("Navigation")

    # Initialize session state if not exists
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = "👥 Team-View"
    
    # Create navigation buttons instead of selectbox
    if st.sidebar.button("⛹🏽‍♂️ Team-View", use_container_width=True):
        st.session_state.analysis_type = "⛹🏽‍♂️ Team-View"
    if st.sidebar.button("🥊 Head-to-Head", use_container_width=True):
        st.session_state.analysis_type = "🥊 Head-to-Head"
    if st.sidebar.button("🏆 Playoff Performance", use_container_width=True):
        st.session_state.analysis_type = "🏆 Playoff Performance"
    if st.sidebar.button("🏅 Medal Overview", use_container_width=True):
        st.session_state.analysis_type = "🏅 Medal Overview"
    if st.sidebar.button("🎯 Drafts", use_container_width=True):
        st.session_state.analysis_type = "🎯 Drafts"
    if st.sidebar.button("👨‍💼 Player Analysis", use_container_width=True):
        st.session_state.analysis_type = "👨‍💼 Player Analysis"
    if st.sidebar.button("📊 Categories", use_container_width=True):
        st.session_state.analysis_type = "📊 Categories"
    if st.sidebar.button("🤝 Trades", use_container_width=True):
        st.session_state.analysis_type = "🤝 Trades"
        
    # Main content based on selection
    if st.session_state.analysis_type == "⛹🏽‍♂️ Team-View":
        # Erstelle die zwei Tabs für Team-View
        tab1, tab2 = st.tabs(["👥 Dashboard", "📜 Historic Drafts"])

        with tab1:
            st.header("Team-View - Manager Dashboard")

            # Überprüfe, ob die Daten geladen wurden
            if seasons_df is not None and not seasons_df.empty:

                # 1. Manager-Dropdown erstellen
                st.subheader("Manager auswählen")

                # Erstelle Liste aller einzigartigen Manager-Namen
                manager_names = sorted(seasons_df['First Name'].dropna().unique())

                # Manager-Dropdown
                selected_manager = st.selectbox(
                "Wählen Sie einen Manager:",
                    options=manager_names,
                    key="team_view_manager_select"
                )

                if selected_manager:
                    st.markdown(f"### Dashboard für **{selected_manager}**")

                    # 2. Filtere Daten für den ausgewählten Manager
                    manager_data = seasons_df[seasons_df['First Name'] == selected_manager].copy()

                    if not manager_data.empty:
                        # 3. Erstelle Tabelle mit den gewünschten Spalten
                        st.subheader("📈 Saison-Historie")

                        # Definiere die gewünschten Spalten
                        table_columns = [
                        'Saison', 'Team Name', 'Wins', 'Losses', 'Ties',
                        'Win-Percentage %', 'Playoff Seed', 'Final Rank'
                        ]

                    # Überprüfe welche Spalten tatsächlich existieren
                    available_columns = [col for col in table_columns if col in manager_data.columns]

                    # Falls 'Saison' nicht existiert, aber 'Year' verfügbar ist, verwende 'Year'
                    if 'Saison' not in available_columns and 'Year' in manager_data.columns:
                        # Ersetze 'Saison' durch 'Year' in der Liste
                        available_columns = ['Year' if col == 'Saison' else col for col in available_columns]
                    
                    if available_columns:
                        # Erstelle Display-Tabelle mit verfügbaren Spalten
                        display_table = manager_data[available_columns].copy()

                        # Bestimme die Jahr-Spalte für die Sortierung
                        year_column = 'Saison' if 'Saison' in display_table.columns else 'Year'
                
                        # Sortiere nach Year absteigend (neueste zuerst)
                        if 'Year' in display_table.columns:
                            display_table = display_table.sort_values('Year', ascending=False)

                        # Benenne die Jahr-Spalte zu 'Saison' um, falls sie 'Year' heißt
                        if 'Year' in display_table.columns and 'Saison' not in display_table.columns:
                            display_table = display_table.rename(columns={'Year': 'Saison'})
                        
                        # Formatiere Win-Percentage als Prozentwert falls vorhanden
                        if 'Win-Percentage %' in display_table.columns:
                            # Falls die Werte als Dezimalzahlen vorliegen (0.75 statt 75)
                            if display_table['Win-Percentage %'].max() <= 1:
                                display_table['Win-Percentage %'] = (display_table['Win-Percentage %'] * 100).round(1)
                            display_table['Win-Percentage %'] = display_table['Win-Percentage %'].astype(str) + '%'

                        # Zeige die Tabelle an
                        st.dataframe(display_table, use_container_width=True, hide_index=True)

                        # 4. Zusätzliche Statistiken
                        st.subheader("📊 Zusammenfassung")

                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            total_seasons = len(manager_data)
                            st.metric("Gespielte Saisons", total_seasons)

                        with col2:
                            if 'Wins' in manager_data.columns:
                                total_wins = manager_data['Wins'].sum()
                                st.metric("Gesamt Siege", int(total_wins))

                        with col3:
                            if 'Losses' in manager_data.columns:
                                total_losses = manager_data['Losses'].sum()
                                st.metric("Gesamt Niederlagen", int(total_losses))

                        with col4:
                            if 'Win-Percentage %' in manager_data.columns and manager_data['Win-Percentage %'].notna().any():
                                # Berechne durchschnittliche Win-Percentage
                                avg_win_pct = manager_data['Win-Percentage %'].mean()
                                if avg_win_pct <= 1:  # Falls als Dezimalzahl
                                    avg_win_pct *= 100
                                st.metric("Ø Win-Rate", f"{avg_win_pct:.1f}%")

                        # 5. Timeline-Grafiken
                        st.subheader("📈 Timelines")

                        # Sortiere Daten für Timeline chronologisch
                        timeline_data = manager_data.copy()
                        if 'Season' in timeline_data.columns:
                            timeline_data = timeline_data.sort_values('Year', ascending=True)

                        # Erstelle zwei Spalten für die Grafiken nebeneinander
                        chart_col1, chart_col2 = st.columns(2)

                        with chart_col1:
                            # Draft Pick Timeline - verwende globale draft_analysis_df Variable
                            st.markdown("**🎯 Draft Pick Timeline**")
                            
                            try:
                                # Filter draft data for the selected manager
                                manager_draft_data = draft_analysis_df[draft_analysis_df['Manager'] == selected_manager]
                                
                                if not manager_draft_data.empty:
                                    
                                    # Try to find the correct year column
                                    if 'Year' in manager_draft_data.columns:
                                        year_col = 'Year'
                                    elif 'Season' in manager_draft_data.columns:
                                        year_col = 'Season'
                                    elif year_columns:
                                        year_col = year_columns[0]
                                    else:
                                        st.error("Keine Jahr-Spalte gefunden!")
                                        year_col = None
                                    
                                    if year_col:
                                        # Sort by year for proper timeline
                                        manager_draft_data = manager_draft_data.sort_values(year_col)
                                        
                                        # Check for draft position column
                                        draft_pos_columns = [col for col in manager_draft_data.columns if 'draft' in col.lower() and 'position' in col.lower()]
                                        if 'Draft_Position' in manager_draft_data.columns:
                                            draft_col = 'Draft_Position'
                                        elif draft_pos_columns:
                                            draft_col = draft_pos_columns[0]
                                        else:
                                            st.error("Keine Draft Position Spalte gefunden!")
                                            draft_col = None
                                        
                                        if draft_col:
                                            
                                            fig_draft = go.Figure()
                                            fig_draft.add_trace(go.Scatter(
                                                x=manager_draft_data[year_col],
                                                y=manager_draft_data[draft_col],
                                                mode='lines+markers',
                                                name='Draft Pick',
                                                line=dict(color='#ff6b6b', width=3),
                                                marker=dict(size=8, color='#ff6b6b')
                                            ))
                                            fig_draft.update_layout(
                                            
                                                xaxis_title='Jahr',
                                                yaxis_title='Draft Pick Position',
                                                yaxis=dict(autorange='reversed'),  # Niedrigere Picks (1, 2, 3) oben
                                                height=400,
                                                showlegend=False
                                            )
                                            st.plotly_chart(fig_draft, use_container_width=True)
                                else:
                                    st.info(f"Keine Draft Pick Daten für {selected_manager} verfügbar")
                                
                            except NameError:
                                st.info("Draft Analysis Daten nicht verfügbar (NameError)")
                            except Exception as e:
                                st.error(f"Fehler beim Laden der Draft Daten: {e}")
                                import traceback
                                st.text(traceback.format_exc())

                        with chart_col2:
                            # Final Rank Timeline
                            if 'Final Rank' in timeline_data.columns and timeline_data['Final Rank'].notna().any():
                                st.markdown("**🏆 Final Rank Timeline**")

                                # Filtere nur Jahre mit Final Rank Daten
                                rank_data = timeline_data[timeline_data['Final Rank'].notna()]

                                # NORMALISIERUNG: Stelle sicher, dass eine Year-Spalte existiert
                                rank_data = normalize_year_column(rank_data)

                                # Sichere Jahr-Spalten-Abfrage
                                def get_year_column_safe(df):
                                    year_columns = ['Year', 'year', 'YEAR', 'Season', 'season', 'SEASON',
                                                    'Saison', 'saison', 'SAISON', 'Date', 'date', 'DATE']

                                    for col in year_columns:
                                        if col in df.columns:
                                            return col
                                    return None

                                year_col = get_year_column_safe(rank_data)

                                if year_col is None:
                                    st.error(f"Keine Jahr-Spalte gefunden. Verfügbare Spalten: {rank_data.columns.tolist()}")
                                else:
                                    # Farbkodierung basierend auf Rank (1-3 = Gold/Silber/Bronze, etc.)
                                    def get_rank_color(rank):
                                        if rank == 1:
                                            return '#FFD700'  # Gold
                                        elif rank == 2:
                                            return '#C0C0C0'  # Silber
                                        elif rank == 3:
                                            return '#CD7F32'  # Bronze
                                        elif rank <= 4:
                                            return '#4CAF50'  # Grün (Playoffs)
                                        elif rank <= 8:
                                            return '#FF9800'  # Orange (Playoffs)
                                        else:
                                            return '#F44336'  # Rot (No Playoffs)

                                    colors = [get_rank_color(rank) for rank in rank_data['Final Rank']]

                                    fig_rank = go.Figure()
                                    fig_rank.add_trace(go.Scatter(
                                        x=rank_data[year_col],  # Verwende die gefundene Jahr-Spalte
                                        y=rank_data['Final Rank'],
                                        mode='lines+markers',
                                        name='Final Rank',
                                        line=dict(color='#2196F3', width=3),
                                        marker=dict(
                                            size=10,
                                            color=colors,
                                            line=dict(color='white', width=2)
                                        )
                                    ))

                                    fig_rank.update_layout(
                                       
                                        xaxis_title='Jahr',
                                        yaxis_title='Final Rank Position',
                                        yaxis=dict(autorange='reversed'),  # Bessere Ranks (1, 2, 3) oben
                                        height=400,
                                        showlegend=False
                                    )

                                    st.plotly_chart(fig_rank, use_container_width=True)

                        # 6. Kombinierte Performance Grafik (wenn beide Daten vorhanden)
                        if ('Final Rank' in timeline_data.columns and
                            'Playoff Seed' in timeline_data.columns and
                            timeline_data['Final Rank'].notna().any() and
                            timeline_data['Playoff Seed'].notna().any()):

                            st.markdown("**📊 Playoff Performance: Playoff-Seed vs. Final Rank**")

                            # Filtere Daten wo beide Werte vorhanden sind
                            playoff_data = timeline_data[
                                (timeline_data['Final Rank'].notna()) &
                                (timeline_data['Playoff Seed'].notna()) &
                                (timeline_data['Playoff Seed'] <= 8)  # Nur echte Playoff Teams
                            ]

                            # Normalisiere die 'Year'-Spalte, bevor sie verwendet wird
                            playoff_data = normalize_year_column(playoff_data)

                            if not playoff_data.empty:
                                fig_performance = go.Figure()

                                # Playoff Seed
                                fig_performance.add_trace(go.Scatter(
                                    x=playoff_data['Year'],
                                    y=playoff_data['Playoff Seed'],
                                    mode='lines+markers',
                                    name='Playoff Seed',
                                    line=dict(color='#9C27B0', width=2, dash='dash'),
                                    marker=dict(size=6, color='#9C27B0')
                                ))

                                # Final Rank
                                fig_performance.add_trace(go.Scatter(
                                    x=playoff_data['Year'],
                                    y=playoff_data['Final Rank'],
                                    mode='lines+markers',
                                    name='Final Rank',
                                    line=dict(color='#FF5722', width=3),
                                    marker=dict(size=8, color='#FF5722')
                                ))

                                fig_performance.update_layout(
                                    
                                    xaxis_title='Jahr',
                                    yaxis_title='Position',
                                    yaxis=dict(autorange='reversed'),
                                    height=400,
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )

                                st.plotly_chart(fig_performance, use_container_width=True)

                                # Kurze Erklärung
                                st.caption("💡 Wenn die rote Linie (Final Rank) unter der violetten Linie (Playoff Seed) liegt = Overperformance (Clutch)")
                            else:
                                st.info("Nicht genügend Playoff-Daten für Vergleich verfügbar")

                    else:
                        st.warning("Die benötigten Spalten wurden im Datensatz nicht gefunden.")
                        st.info("Verfügbare Spalten: " + ", ".join(manager_data.columns.tolist()))

                else:
                    st.warning(f"Keine Daten für Manager '{selected_manager}' gefunden.")

            else:
                st.warning("Die Seasons-Daten konnten nicht geladen werden.")

        with tab2:
            st.header("Historic Drafts - Manager Draft Historie")

            # Überprüfe, ob die Draft-Daten geladen wurden
            if drafts_df is not None and not drafts_df.empty:

                # 1. Manager-Dropdown erstellen (identisch zum ersten Tab)
                st.subheader("Manager auswählen")

                # Erstelle Liste aller einzigartigen Manager-Namen aus drafts_df
                manager_names = sorted(drafts_df['Manager'].dropna().unique()) if 'Manager' in drafts_df.columns else []

                if not manager_names:
                    st.error("Keine Manager in den Draft-Daten gefunden. Überprüfen Sie die 'Manager' Spalte in drafts_df.")
                else:
                    # Manager-Dropdown
                    selected_manager = st.selectbox(
                        "Wählen Sie einen Manager:",
                        options=manager_names,
                        key="historic_drafts_manager_select"
                    )

                if selected_manager:
                    st.markdown(f"### Draft-Historie für **{selected_manager}**")

                    # 2. Filtere Draft-Daten für den ausgewählten Manager
                    manager_drafts = drafts_df[drafts_df['Manager'] == selected_manager].copy()

                    if not manager_drafts.empty:
                        # 3. Sortiere nach Jahr absteigend (neueste zuerst)
                        year_col = None
                        for col in ['Year', 'Season', 'Jahr', 'Saison']:
                            if col in manager_drafts.columns:
                                year_col = col
                                break

                        if year_col:
                            manager_drafts = manager_drafts.sort_values(year_col, ascending=False)
                            years = sorted(manager_drafts[year_col].unique(), reverse=True)
                        else:
                            st.warning("Keine Jahr-Spalte gefunden in den Draft-Daten")
                            years = ['Alle Jahre']

                        # 4. Erstelle Draft-Übersicht für jede Saison
                        st.subheader("🎯 Draft-Übersicht nach Saisons")

                        # Organisiere Jahre für 2-spaltiges Layout
                        years_pairs = []
                        for i in range(0, len(years), 2):
                            if i + 1 < len(years):
                                years_pairs.append((years[i], years[i + 1]))
                            else:
                                years_pairs.append((years[i], None))

                        for year_pair in years_pairs:
                            year1, year2 = year_pair
                    
                            # Erstelle 2 Spalten für nebeneinander liegende Tabellen
                            col1, col2 = st.columns(2)
                    
                            # Erste Saison (linke Spalte)
                            with col1:
                                _display_season_draft(manager_drafts, year1, year_col)
                    
                            # Zweite Saison (rechte Spalte), falls vorhanden
                            with col2:
                                if year2:
                                    _display_season_draft(manager_drafts, year2, year_col)

                    else:
                        st.warning(f"Keine Draft-Daten für Manager '{selected_manager}' gefunden.")

                else:
                    st.warning("Die Draft-Daten konnten nicht geladen werden. Überprüfen Sie drafts_df.")
            
    elif st.session_state.analysis_type == "🥊 Head-to-Head":
        st.header("Head-to-Head Analysis")
        
        managers = sorted(teams_df['First Name'].unique())
        
        # Tabs for different H2H analyses
        tab1, tab2 = st.tabs(["🎯 Lieblings- & Angstgegner", "💥 Direkter Vergleich"])
        
        with tab1:
            st.subheader("Lieblings- & Angstgegner Analyse")
            st.markdown("*Mindestanzahl: 5 Spiele gegeneinander*")
            
            # Manager selection
            selected_manager = st.selectbox(
                "Wähle Manager für Analyse:", 
                managers, 
                key="opponent_analysis_manager"
            )
            
            # Calculate and display opponent analysis
            favorites, nightmares = calculate_all_h2h_stats(processed_df, selected_manager, min_games=5)
            
            st.markdown(f"### Analyse für **{selected_manager}**")
            display_opponent_analysis(favorites, nightmares, selected_manager)
            
            # Show all H2H stats table
            st.markdown("### Alle Head-to-Head Statistiken")
            all_opponents = []
            all_managers_set = set(processed_df['Home_Manager'].unique()) | set(processed_df['Away_Manager'].unique())
            all_managers_set.discard(selected_manager)
            
            for opponent in all_managers_set:
                h2h_stats = calculate_head_to_head(processed_df, selected_manager, opponent)
                if h2h_stats['games'] >= 5:
                    all_opponents.append({
                        'Gegner': opponent,
                        'Spiele': h2h_stats['games'],
                        'Siege': h2h_stats['wins'],
                        'Niederlagen': h2h_stats['losses'],
                        'Unentschieden': h2h_stats['ties'],
                        'Siegquote': f"{h2h_stats['win_pct']:.1%}"
                    })
            
            if all_opponents:
                all_h2h_df = pd.DataFrame(all_opponents).sort_values('Spiele', ascending=False)
                st.dataframe(all_h2h_df, use_container_width=True, hide_index=True)
            else:
                st.info("Keine Gegner mit mindestens 5 Spielen gefunden.")
        
        with tab2:
            st.subheader("Direkter Manager Vergleich")
            
            col1, col2 = st.columns(2)
            with col1:
                manager1 = st.selectbox("Select Manager 1", managers, index=0, key="manager1")
            with col2:
                manager2 = st.selectbox("Select Manager 2", managers, index=1 if len(managers) > 1 else 0, key="manager2")
            
            if manager1 != manager2:
                h2h_stats = calculate_head_to_head(processed_df, manager1, manager2)
                
                if h2h_stats and h2h_stats['games'] > 0:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Total Games", h2h_stats['games'])
                    with col2:
                        st.metric(f"{manager1} Wins", h2h_stats['wins'])
                    with col3:
                        st.metric(f"{manager2} Wins", h2h_stats['losses'])
                    with col4:
                        st.metric("Ties", h2h_stats['ties'])
                    with col5:
                        st.metric(f"{manager1} Win %", f"{h2h_stats['win_pct']:.1%}")
                    
                    # Visualization
                    fig = go.Figure(data=[
                        go.Bar(name=manager1, x=[manager1], y=[h2h_stats['wins']], marker_color='#1f77b4'),
                        go.Bar(name=manager2, x=[manager2], y=[h2h_stats['losses']], marker_color='#ff7f0e')
                    ])
                    fig.update_layout(title=f"{manager1} vs {manager2} - Head to Head")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No matchups found between {manager1} and {manager2}")
            else:
                st.info("Please select two different managers.")

    elif st.session_state.analysis_type == "🏆 Playoff Performance":
        st.header("Playoff Performance Analysis")
        
        # Create tabs
        tab1, tab2 = st.tabs(["📊 Performance Overview", "😱 Choke vs Clutch 💪"])
        
        with tab1:
            # Calculate stats
            full_stats, reg_ranked, playoff_ranked = calculate_playoff_stats(processed_df, teams_df)
            
            if full_stats is not None:
                # Rankings side by side
                st.subheader("Rankings: Who performs when it matters?")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**🏀 Best Regular Season Teams**")
                    reg_styled = style_dataframe_with_colors(reg_ranked, ['Regular Win%'])
                    st.dataframe(reg_styled, use_container_width=True, hide_index=True)
                with col2:
                    st.markdown("**🔥 Best Playoff Teams**")
                    playoff_styled = style_dataframe_with_colors(playoff_ranked, ['Playoff Win%'])
                    st.dataframe(playoff_styled, use_container_width=True, hide_index=True)
                # Full overview table
                st.subheader("Complete Overview: Regular vs Playoff Performance")
                full_styled = style_dataframe_with_colors(full_stats, ['Regular Win%', 'Playoff Win%'])
                st.dataframe(full_styled, use_container_width=True, hide_index=True)
        
        with tab2:
            st.subheader("😱 Choke vs Clutch 💪 Analysis")
            
            # Check if seasons_df is available
            if 'seasons_df' in locals() or 'seasons_df' in globals():
                # Get all manager names
                manager_names = sorted(seasons_df['First Name'].dropna().unique())
                
                # Calculate choke/clutch stats for each manager
                choke_stats = []
                clutch_stats = []
                
                for manager in manager_names:
                    # Filter data for this manager
                    manager_data = seasons_df[seasons_df['First Name'] == manager].copy()
                    
                    # Remove rows where Playoff Seed or Final Rank is NaN and filter for playoff seeds 1-8 only
                    valid_data = manager_data.dropna(subset=['Playoff Seed', 'Final Rank'])
                    valid_data = valid_data[valid_data['Playoff Seed'] <= 8]
                    
                    if not valid_data.empty:
                        # Calculate differences (Final Rank - Playoff Seed)
                        # Positive difference = worse final rank than seed = Choke
                        # Negative difference = better final rank than seed = Clutch
                        valid_data['Difference'] = valid_data['Final Rank'] - valid_data['Playoff Seed']
                        
                        # Count chokes and clutches
                        chokes = len(valid_data[valid_data['Difference'] > 0])
                        clutches = len(valid_data[valid_data['Difference'] < 0])
                        neutral = len(valid_data[valid_data['Difference'] == 0])
                        total_playoff_appearances = len(valid_data)
                        
                        # Calculate average difference
                        avg_difference = valid_data['Difference'].mean()
                        
                        # Calculate total sum of all chokes and clutches
                        total_choke_sum = valid_data[valid_data['Difference'] > 0]['Difference'].sum()
                        total_clutch_sum = valid_data[valid_data['Difference'] < 0]['Difference'].sum()
                        
                        # Add to appropriate list based on overall tendency
                        if avg_difference > 0:  # More choking than clutching
                            choke_stats.append({
                                'Manager': manager,
                                'Total Sum': -total_choke_sum,  # Negative for total choking
                                'Chokes': chokes,
                                'Clutches': clutches,
                                'Neutral': neutral,
                                'Choking Index': round(-avg_difference, 2)  # Negative for choking
                            })
                        else:  # More clutching than choking
                            clutch_stats.append({
                                'Manager': manager,
                                'Total Sum': abs(total_clutch_sum),  # Positive for total clutching
                                'Clutches': clutches,
                                'Chokes': chokes,
                                'Neutral': neutral,
                                'Clutch-O-Meter': round(abs(avg_difference), 2)  # Positive for clutching
                            })
                
                # Create DataFrames
                if choke_stats:
                    choke_df = pd.DataFrame(choke_stats)
                    choke_df = choke_df.sort_values('Total Sum', ascending=True)  # Most negative first
                
                if clutch_stats:
                    clutch_df = pd.DataFrame(clutch_stats)
                    clutch_df = clutch_df.sort_values('Total Sum', ascending=False)  # Highest positive first
                
                # Display tables side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 😱 Choking Index")
                    st.markdown("*Managers who underperform in playoffs*")
                    
                    if choke_stats:
                        # Style the choke table with red colors
                        def style_choke_table(df):
                            def highlight_chokes(val):
                                if isinstance(val, (int, float)):
                                    if val < -6:  # Very negative = very bad choking
                                        return 'background-color: #ffebee; color: #c62828; font-weight: bold'
                                    elif val < -3:
                                        return 'background-color: #ffcdd2; color: #d32f2f'
                                    elif val < 0:
                                        return 'background-color: #ffecb3; color: #ef6c00'
                                return ''
                            
                            styled = df.style.applymap(highlight_chokes, subset=['Choking Index', 'Total Sum'])
                            styled = styled.applymap(lambda x: 'background-color: #ffebee' if x > df['Chokes'].mean() else '', subset=['Chokes'])
                            return styled
                        
                        choke_styled = style_choke_table(choke_df)
                        st.dataframe(
                            choke_styled,
                            column_config={
                                "Manager": "Manager",
                                "Total Sum": st.column_config.NumberColumn("😱 Index", help="Summe aller Chokes (negativer = schlechter)", format="%.0f"),
                                "Chokes": st.column_config.NumberColumn("🔴 Choke", help="Anzahl der Underperformances"),
                                "Clutches": st.column_config.NumberColumn("🟢 Clutch", help="Anzahl der Overperformances"),
                                "Neutral": st.column_config.NumberColumn("⚪ Neutral", help="Performances wie erwartet"),
                                "Choking Index": st.column_config.NumberColumn("Durchschnitt", help="Durchschnittliche Underperformance (negativer = schlechter)", format="%.2f")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.success("No chokers found! 🎉")
                
                with col2:
                    st.markdown("### 🔥 Clutch-O-Meter")
                    st.markdown("*Managers who rise to the occasion*")
                    
                    if clutch_stats:
                        # Style the clutch table with green colors
                        def style_clutch_table(df):
                            def highlight_clutches(val):
                                if isinstance(val, (int, float)):
                                    if val > 6:  # Very positive = very good clutching
                                        return 'background-color: #e8f5e8; color: #2e7d32; font-weight: bold'
                                    elif val > 3:
                                        return 'background-color: #c8e6c9; color: #388e3c'
                                    elif val > 0:
                                        return 'background-color: #dcedc8; color: #689f38'
                                return ''
                            
                            styled = df.style.applymap(highlight_clutches, subset=['Clutch-O-Meter', 'Total Sum'])
                            styled = styled.applymap(lambda x: 'background-color: #e8f5e8' if x > df['Clutches'].mean() else '', subset=['Clutches'])
                            return styled
                        
                        clutch_styled = style_clutch_table(clutch_df)
                        st.dataframe(
                            clutch_styled,
                            column_config={
                                "Manager": "Manager",
                                "Total Sum": st.column_config.NumberColumn("💪 Clutch-O-Meter", help="Summe aller Clutches (höher = besser)", format="%.0f"),
                                "Clutches": st.column_config.NumberColumn("🟢 Clutch", help="Anzahl der Overperformances"), 
                                "Chokes": st.column_config.NumberColumn("🔴 Choke", help="Anzahl der Underperformances"),
                                "Neutral": st.column_config.NumberColumn("⚪ Neutral", help="Performances wie erwartet"),
                                "Clutch-O-Meter": st.column_config.NumberColumn("Durchschnitt", help="Durchschnittliche Overperformance (höher = besser)", format="%.2f")
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No clutch performers found! 😅")
                
                # Explanation
                st.markdown("---")
                st.markdown("""
                **How it works:**
                - **Only Playoff Seeds 1-8 are considered** (teams that made playoffs)
                - **Choke**: Final Rank > Playoff Seed (e.g., 3rd seed finishes 5th = Choke)
                - **Clutch**: Final Rank < Playoff Seed (e.g., 6th seed finishes 2nd = Clutch)
                - **Choking Index**: Durchschnitt aller Chokes (NEGATIV = schlechter)
                  - Beispiel: -2.5 = Du verlierst durchschnittlich 2.5 Plätze
                - **Total Sum**: Summe aller Differenzen (NEGATIV bei Chokes, POSITIV bei Clutches)
                  - Beispiel Choke: 2x(-2) + 1x(-1) = -5 total
                  - Beispiel Clutch: 2x(+3) + 1x(+1) = +7 total
                - **Clutch-O-Meter**: Durchschnitt aller Clutches (POSITIV = besser)
                  - Beispiel: 1.8 = Du gewinnst durchschnittlich 1.8 Plätze
                - **Color Coding**: 
                  - 🔴 Red shades = Choking (je dunkler, desto schlechter)
                  - 🟢 Green shades = Clutching (je dunkler, desto besser)
                """)
                
            else:
                st.error("seasons_df ist nicht verfügbar. Bitte stellen Sie sicher, dass die Daten geladen wurden.")
        
    elif st.session_state.analysis_type == "🏅 Medal Overview":
        st.header("Medal Overview")
    
        # Create tabs
        tab1, tab2 = st.tabs(["🏆 Medal Table", "📊 Ewige Tabelle"])
        
        with tab1:
            medal_table = create_medal_table(teams_df)
            
            if medal_table is not None:
                st.subheader("🏆 Medal Table")
                
                # Display medal table
                medal_styled = medal_table.style.applymap(lambda x: "font-weight: bold;", subset=['Manager'])
                st.dataframe(
                    medal_styled,
                    column_config={
                        "Rank": "Rank",
                        "Manager": "Manager",
                        "Gold": "🥇 Gold",
                        "Silver": "🥈 Silver", 
                        "Bronze": "🥉 Bronze",
                        "Total": "Total"
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Medal visualization
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='🥇 Gold',
                    x=medal_table['Manager'],
                    y=medal_table['Gold'],
                    marker_color='#FFD700'  # Gold color
                ))
                
                fig.add_trace(go.Bar(
                    name='🥈 Silver',
                    x=medal_table['Manager'],
                    y=medal_table['Silver'],
                    marker_color='#C0C0C0'  # Silver color
                ))
                
                fig.add_trace(go.Bar(
                    name='🥉 Bronze',
                    x=medal_table['Manager'],
                    y=medal_table['Bronze'],
                    marker_color='#CD7F32'  # Bronze color
                ))
                
                fig.update_layout(
                    title='Medal Distribution by Manager',
                    xaxis_title='Manager',
                    yaxis_title='Number of Medals',
                    barmode='stack',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Ewige Tabelle")
            st.write("""
            Sortiert nach den meisten Siegen. Bei Gleichstand wird die Win-Percentage % verglichen.
            Notiz: 2 Ties ergeben 1 Win (Default Berechnung von ESPN)
            """)

        # Create eternal table from seasons_df
        if 'seasons_df' in locals() or 'seasons_df' in globals():
            # Now we know the correct column names: Year, Wins, Losses, Ties, First Name
            if all(col in seasons_df.columns for col in ['First Name', 'Wins', 'Losses', 'Ties', 'Saison']):
                # Group by First Name and calculate statistics
                eternal_stats = seasons_df.groupby('First Name').agg({
                    'Wins': 'sum',
                    'Losses': 'sum', 
                    'Ties': 'sum',
                    'Saison': 'count'  # Count of seasons played
                }).reset_index()
            
                # Rename columns for consistency
                eternal_stats = eternal_stats.rename(columns={
                    'First Name': 'Manager',
                    'Saison': 'Gespielte Saisons'
                })
                        
                # Berechnung der Gesamtzahl der Spiele
                total_games = eternal_stats['Wins'] + eternal_stats['Losses'] + eternal_stats['Ties']

                # Berechne die Anzahl der "equivalent wins" (Siege + die Hälfte der Unentschieden)
                equivalent_wins = eternal_stats['Wins'] + (eternal_stats['Ties'] / 2)

                # Berechne die neue Sieg-Quote
                eternal_stats['Win-Percentage %'] = (equivalent_wins / total_games * 100).round(2)
            
                # Sort by: 1. Most Wins (descending), 2. Highest Win-Percentage (descending)
                eternal_stats = eternal_stats.sort_values(['Wins', 'Win-Percentage %'], ascending=[False, False])
            
                # Add ranking
                eternal_stats['Ranking'] = range(1, len(eternal_stats) + 1)
            
                # Reorder columns
                eternal_stats = eternal_stats[['Ranking', 'Manager', 'Wins', 'Losses', 'Ties', 'Win-Percentage %', 'Gespielte Saisons']]
            
                # Display the eternal table
                st.dataframe(
                    eternal_stats,
                    column_config={
                        "Ranking": "Ranking",
                        "Manager": "Manager",
                        "Wins": "Wins",
                        "Losses": "Losses",
                        "Ties": "Ties", 
                        "Win-Percentage %": "Win-Percentage %",
                        "Gespielte Saisons": "Gespielte Saisons"
                    },
                    hide_index=True,
                    use_container_width=True
                )
            
                # Visualization for eternal table
                fig_eternal = go.Figure()
            
                fig_eternal.add_trace(go.Bar(
                    name='Wins',
                    x=eternal_stats['Manager'],
                    y=eternal_stats['Wins'],
                    marker_color='#28a745'  # Green for wins
                ))
            
                fig_eternal.add_trace(go.Bar(
                    name='Losses',
                    x=eternal_stats['Manager'],
                    y=eternal_stats['Losses'],
                    marker_color='#dc3545'  # Red for losses
                ))
            
                fig_eternal.add_trace(go.Bar(
                    name='Ties',
                    x=eternal_stats['Manager'],
                    y=eternal_stats['Ties'],
                    marker_color='#ffc107'  # Yellow for ties
                ))
            
                fig_eternal.update_layout(
                    title='Ewige Tabelle - Wins/Losses/Ties Distribution',
                    xaxis_title='Manager',
                    yaxis_title='Number of Games',
                    barmode='stack',
                    height=500
                )
            
                st.plotly_chart(fig_eternal, use_container_width=True)
            
            else:
                st.error("Required columns not found in seasons_df. Please check the data.")
            
        else:
            st.error("seasons_df ist nicht verfügbar. Bitte stellen Sie sicher, dass die Daten geladen wurden.")
    
    elif st.session_state.analysis_type == "🎯 Drafts":
        st.header("Draft Analysis")
        
        
        if draft_analysis_df is not None and not draft_analysis_df.empty:
            # Tabs for different draft analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚀 Over/Under Performance", "📊 Draft vs Final Position", "🍀 Lottery Luck", "🎯 Manager Performance", "📈 Draft Value Analysis"])
            
            with tab1:
                st.subheader("Over/Under Performance")
                st.markdown("*Diskrepanz zwischen Draft-Position und finalem Rang*")
                
                # Season filter
                seasons = sorted(draft_analysis_df['Season'].unique(), reverse=True)
                selected_season = st.selectbox("Saison auswählen:", ["Alle Saisons"] + list(seasons))
                if selected_season != "Alle Saisons":
                    filtered_df = draft_analysis_df[draft_analysis_df['Season'] == selected_season]
                else:
                    filtered_df = draft_analysis_df
                
                # Top Over/Underperformers
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🚀 Beste Overperformer")
                    st.markdown("*Höchste positive Abweichung (Draft schlechter als Endrang)*")
                    best_over = filtered_df.nlargest(5, 'Over_Under')
                    
                    for i, (_, row) in enumerate(best_over.iterrows()):
                        st.markdown(f"""
                        <div class="favorite-opponent">
                            <h4>#{i+1} {row['Manager']} ({row['Season']})</h4>
                            <p><strong>+{row['Over_Under']}</strong> (Pick {row['Draft_Position']} → Rang {row['Final_Rank']})</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 😰 Größte Underperformer")  
                    st.markdown("*Höchste negative Abweichung (Draft besser als Endrang)*")
                    worst_under = filtered_df.nsmallest(5, 'Over_Under')
                    
                    for i, (_, row) in enumerate(worst_under.iterrows()):
                        st.markdown(f"""
                        <div class="nightmare-opponent">
                            <h4>#{i+1} {row['Manager']} ({row['Season']})</h4>
                            <p><strong>{row['Over_Under']}</strong> (Pick {row['Draft_Position']} → Rang {row['Final_Rank']})</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with tab2:
                st.subheader("Draft Position vs Final Rank")
                
                # Scatter plot
                fig = create_draft_scatter_plot(draft_analysis_df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Correlation analysis
                correlation, p_value = calculate_correlation(
                    draft_analysis_df['Draft_Position'], 
                    draft_analysis_df['Final_Rank']
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Correlation", f"{correlation:.3f}")
                with col2:
                    st.metric("P-Value", f"{p_value:.3f}")
                with col3:
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    st.metric("Statistical Significance", significance)
                
                st.markdown(f"""
                **Interpretation:**
                - Correlation of {correlation:.3f} indicates {"a strong" if abs(correlation) > 0.7 else "a moderate" if abs(correlation) > 0.3 else "a weak"} relationship
                - {"Draft position is a good predictor of final rank" if abs(correlation) > 0.5 else "Draft position has limited predictive power"}
                """)
            
            with tab3:
                st.subheader("Lottery Luck")
                st.markdown("*Welcher Manager hatte in wie viel Prozent seiner Saisons einen Top 3 Pick?*")
                
                # Calculate lottery luck for each manager
                lottery_stats = []
                managers = draft_analysis_df['Manager'].unique()
                
                for manager in managers:
                    manager_data = draft_analysis_df[draft_analysis_df['Manager'] == manager]
                    total_seasons = len(manager_data)
                    top3_picks = len(manager_data[manager_data['Draft_Position'] <= 3])
                    top3_percentage = (top3_picks / total_seasons * 100) if total_seasons > 0 else 0
                    
                    lottery_stats.append({
                        'Manager': manager,
                        'Total_Seasons': total_seasons,
                        'Top3_Picks': top3_picks,
                        'Top3_Percentage': round(top3_percentage, 1)
                    })
                
                # Create DataFrame and sort by percentage
                lottery_df = pd.DataFrame(lottery_stats).sort_values('Top3_Percentage', ascending=False)

                # --- FÜGE DIESE ZEILE HINZU, UM NACH SAISONS ZU FILTERN ---
                lottery_df = lottery_df[lottery_df['Total_Seasons'] >= 5]
                
                # Display lottery luck table
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Color coding for lottery luck
                    def highlight_lottery_luck(val):
                        if pd.isna(val):
                            return ""
                        if val >= 40:
                            return "background-color: rgba(0, 150, 0, 0.4);"  # Very lucky - Dark green
                        elif val >= 30:
                            return "background-color: rgba(50, 180, 50, 0.3);"  # Lucky - Medium green
                        elif val >= 20:
                            return "background-color: rgba(150, 220, 150, 0.2);"  # Above average - Light green
                        elif val >= 10:
                            return "background-color: rgba(255, 255, 0, 0.1);"  # Average - Light yellow
                        else:
                            return "background-color: rgba(255, 150, 100, 0.3);"  # Unlucky - Orange
                    
                    styled_lottery = lottery_df.style.applymap(
                        highlight_lottery_luck, 
                        subset=['Top3_Percentage']
                    )
                    
                    st.dataframe(
                        styled_lottery,
                        column_config={
                            "Manager": "Manager",
                            "Total_Seasons": "Gespielte Saisons",
                            "Top3_Picks": "Top 3 Picks",
                            "Top3_Percentage": "Top 3 Pick %"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                
                with col2:
                    # Show top 3 most and least lucky managers
                    st.markdown("**🍀 Glückspilze**")
                    top_lucky = lottery_df.head(3)
                    for i, (_, row) in enumerate(top_lucky.iterrows()):
                        st.markdown(f"**{i+1}.** {row['Manager']} ({row['Top3_Percentage']}%)")
                    
                    st.markdown("**😔 Pechvögel**")
                    bottom_lucky = lottery_df.tail(3).iloc[::-1]  # Reverse order
                    for i, (_, row) in enumerate(bottom_lucky.iterrows()):
                        st.markdown(f"**{i+1}.** {row['Manager']} ({row['Top3_Percentage']}%)")
                
                # Visualization
                fig_lottery = px.bar(
                    lottery_df,
                    x='Manager',
                    y='Top3_Percentage',
                    title='Lottery Luck: Top 3 Pick Percentage by Manager',
                    labels={'Top3_Percentage': 'Top 3 Pick %', 'Manager': 'Manager'},
                    color='Top3_Percentage',
                    color_continuous_scale='RdYlGn',
                    template="plotly_dark"
                )
                fig_lottery.update_layout(height=400, showlegend=False)
                fig_lottery.add_hline(y=33.3, line_dash="dash", line_color="gray", annotation_text="Erwartungswert (33.3%)")
                st.plotly_chart(fig_lottery, use_container_width=True)
            
            with tab4:
                st.subheader("Manager Draft Performance")
                
                # Cumulative over/under analysis
                cumulative_df = calculate_cumulative_over_under(draft_analysis_df)
                
                if cumulative_df is not None:
                    st.markdown("**Over/Under Performance (Positive = Overperformed)**")
                    
                    # Round the average columns to whole numbers
                    display_cumulative = cumulative_df.copy()
                    display_cumulative['Avg_Draft_Position'] = display_cumulative['Avg_Draft_Position'].round(0).astype(int)
                    display_cumulative['Avg_Final_Rank'] = display_cumulative['Avg_Final_Rank'].round(0).astype(int)
                    
                    # Color coding for over/under performance
                    def highlight_over_under(val):
                        if pd.isna(val):
                            return ""
                        if val > 5:
                            return "background-color: rgba(0, 150, 0, 0.4);"  # Dark green
                        elif val > 2:
                            return "background-color: rgba(50, 180, 50, 0.3);"  # Medium green
                        elif val > 0:
                            return "background-color: rgba(150, 220, 150, 0.2);"  # Light green
                        elif val == 0:
                            return "background-color: rgba(255, 255, 0, 0.1);"  # Light yellow
                        elif val > -2:
                            return "background-color: rgba(255, 200, 150, 0.2);"  # Light orange
                        elif val > -5:
                            return "background-color: rgba(255, 150, 100, 0.3);"  # Medium orange
                        else:
                            return "background-color: rgba(200, 0, 0, 0.4);"  # Dark red
                    
                    styled_cumulative = display_cumulative.style.applymap(
                        highlight_over_under, 
                        subset=['Kumulierter_Over_Under']
                    )
                    
                    st.dataframe(
                        styled_cumulative,
                        column_config={
                            "Manager": "Manager",
                            "Kumulierter_Over_Under": "Kumulierter Over/Under",
                            "Anzahl_Saisons": "Anzahl Saisons",
                            "Avg_Draft_Position": "Ø Draft Position",
                            "Avg_Final_Rank": "Ø Final Rank"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Bar chart of cumulative performance
                    fig = px.bar(
                        cumulative_df,
                        x='Manager',
                        y='Kumulierter_Over_Under',
                        title='Kumulative Over/Under Performance by Manager',
                        color='Kumulierter_Over_Under',
                        color_continuous_scale='RdYlGn',
                        template="plotly_dark"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab5:
                st.subheader("Draft Value Analysis")
                
                # Calculate draft value
                draft_value_df = calculate_draft_value_analysis(draft_analysis_df)
                
                if draft_value_df is not None:
                    st.markdown("**Average Final Rank by Draft Position**")
                    
                    # Round the average final rank to whole numbers and remove unnecessary columns
                    display_df = draft_value_df[['Draft_Position', 'Avg_Final_Rank']].copy()
                    display_df['Avg_Final_Rank'] = display_df['Avg_Final_Rank'].round(0).astype(int)
                    
                    st.dataframe(
                        display_df,
                        column_config={
                            "Draft_Position": "Draft Position",
                            "Avg_Final_Rank": "Ø Final Rank"
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Line chart showing expected vs actual performance
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=draft_value_df['Draft_Position'],
                        y=draft_value_df['Avg_Final_Rank'],
                        mode='lines+markers',
                        name='Actual Performance',
                        line=dict(color='blue')
                    ))
                    
                    # Perfect prediction line
                    fig.add_trace(go.Scatter(
                        x=draft_value_df['Draft_Position'],
                        y=draft_value_df['Draft_Position'],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig.update_layout(
                        title='Draft Position vs Average Final Rank',
                        xaxis_title='Draft Position',
                        yaxis_title='Average Final Rank',
                        yaxis=dict(autorange='reversed'),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Draft data not available or could not be processed. Please check your mDrafts sheet URL and data format.")

    elif st.session_state.analysis_type == "👨‍💼 Player Analysis":
        st.header("Player Analysis")
        
        # Tabs for different player analyses
        tab1, tab2, tab3 = st.tabs(["🏆 Championship DNA", "🎯 Legenden", "📊 Manager-Player Loyalty"])
        
        with tab1:
            st.subheader("Championship DNA")
            st.markdown("*Welche Spieler standen am häufigsten in Championship-Teams?*")
            
            # Calculate championship DNA
            champ_df, finals_df, contender_df = calculate_championship_dna(drafts_df, teams_df)
          
            if champ_df is not None and 'Championship_Years' in champ_df.columns:
                def fix_championship_years(x):
                    if pd.isna(x) or x == '' or str(x).lower() == 'nan':
                        return x
                    try:
                        years = str(x).split(', ')
                        fixed_years = []
                        for year in years:
                            year = year.strip()
                            if year and year.replace('.', '').replace('-', '').isdigit():
                                try:
                                    fixed_years.append(str(int(float(year))))
                                except (ValueError, TypeError):
                                    fixed_years.append(year)
                            elif year:
                                fixed_years.append(year)
                        return ', '.join(fixed_years) if fixed_years else str(x)
                    except Exception:
                        return str(x)
                
                champ_df['Championship_Years'] = champ_df['Championship_Years'].apply(fix_championship_years)
                
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🏆 Championship Players")
                st.markdown("*Häufigste Meister-Team Mitglieder*")
                
                if champ_df is not None and len(champ_df) > 0:
                    for i, (_, player) in enumerate(champ_df.head(10).iterrows()):
                        st.markdown(f"""
                        <div class="champion-player">
                            <h4>#{i+1} {player['Player']}</h4>
                            <p><strong>{player['Championships']}</strong> Championships ({player['Championship_Years']})</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Keine Championship-Daten verfügbar. Diese werden mit echten Spieler-Roster-Daten gefüllt.")
            
            with col2:
                st.markdown("### 🥈 Finals Appearances")
                st.markdown("*Spieler mit den meisten Finals-Teilnahmen*")
                
                if finals_df is not None and len(finals_df) > 0:
                    for i, (_, player) in enumerate(finals_df.head(10).iterrows()):
                        win_rate_display = f"{player['Finals_Win_Rate']:.1%}" if player['Finals_Win_Rate'] > 0 else "0%"
                        st.markdown(f"""
                        <div class="champion-player">
                            <h4>#{i+1} {player['Player']}</h4>
                            <p><strong>{player['Finals_Appearances']}</strong> Finals ({player['Championships']} Siege) - {win_rate_display} Win Rate</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Keine Finals-Daten verfügbar. Diese werden mit echten Spieler-Roster-Daten gefüllt.")
            
            # Championship visualization
            if champ_df is not None:
                fig = px.bar(
                    champ_df.head(15),
                    x='Player',
                    y='Championships',
                    title='Top 15 Championship Players',
                    color='Championships',
                    color_continuous_scale='Reds',
                    template="plotly_dark"
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Legenden")
            
            # Calculate legend analysis
            first_round_df, playoff_heroes_df = calculate_legend_analysis(drafts_df, teams_df, contender_df)
            
            # FIX: Convert year ranges like "2015.0-2025.0" to "2015-2025"
            if first_round_df is not None and 'Years_as_Superstar' in first_round_df.columns:
                def fix_years_column_safe(x):
                    try:
                        # Handle various None/NaN cases
                        if x is None or pd.isna(x):
                            return x
                        
                        x_str = str(x)
                        if x_str.lower() in ['nan', 'none', '']:
                            return x
                        
                        # If it's already a clean string without decimals, return as is
                        if '.' not in x_str:
                            return x_str
                        
                        # Handle both comma-separated and dash-separated years
                        if '-' in x_str:
                            # Handle ranges like "2015.0-2025.0"
                            years = x_str.split('-')
                        else:
                            # Handle comma-separated like "2015.0, 2016.0"
                            years = x_str.split(', ')
                        
                        fixed_years = []
                        for year in years:
                            year = year.strip()
                            if not year:
                                continue
                                
                            # Try to convert decimal years to integers
                            try:
                                if '.' in year and year.replace('.', '').isdigit():
                                    fixed_years.append(str(int(float(year))))
                                else:
                                    fixed_years.append(year)
                            except (ValueError, TypeError):
                                fixed_years.append(year)
                        
                        # Rejoin with appropriate separator
                        separator = '-' if '-' in x_str else ', '
                        return separator.join(fixed_years) if fixed_years else str(x)
                    
                    except Exception:
                        return str(x)
                
                first_round_df['Years_as_Superstar'] = first_round_df['Years_as_Superstar'].apply(fix_years_column_safe)
            
                
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ⭐ 1st Round Superstars")
                st.markdown("*Am häufigsten in Runde 1 gedraftet*")
                
                if first_round_df is not None and len(first_round_df) > 0:
                    for i, (_, player) in enumerate(first_round_df.head(8).iterrows()):
                        st.markdown(f"""
                        <div class="legend-player">
                            <h4>#{i+1} {player['Player']}</h4>
                            <p><strong>{player['First_Round_Picks']}</strong> First Round Picks</p>
                            <p>Ø Pick {player['Avg_Pick_in_Round1']} | {player['Years_as_Superstar']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Visualization
                if first_round_df is not None:
                    fig1 = px.bar(
                        first_round_df.head(10),
                        x='Player',
                        y='First_Round_Picks',
                        title='Top 10 First Round Superstars',
                        color='Avg_Pick_in_Round1',
                        color_continuous_scale='Blues_r',
                        template="plotly_dark"
                    )
                    fig1.update_layout(height=350, xaxis_tickangle=-45)
                    st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown("### 🔥 Playoff Heroes")
                st.markdown("*Überproportional oft in Playoff-Teams*")
                
                if playoff_heroes_df is not None and len(playoff_heroes_df) > 0:
                    for i, (_, player) in enumerate(playoff_heroes_df.head(8).iterrows()):
                        st.markdown(f"""
                        <div class="legend-player">
                            <h4>#{i+1} {player['Player']}</h4>
                            <p><strong>{player['Playoff_Hero_Seasons']}</strong> Playoff_Hero_Seasons</p>
                            <p>Ø Pick {player['Avg_Draft_Position']} | {player['Playoff_Rate']:.0%} Rate | Score: {player['Hidden_Gem_Score']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Visualization
                if playoff_heroes_df is not None:
                    fig2 = px.scatter(
                        playoff_heroes_df,
                        x='Avg_Draft_Position',
                        y='Playoff_Rate',
                        size='Playoff_Appearances',
                        color='Hidden_Gem_Score',
                        hover_data=['Player'],
                        title='Playoff Heroes: Draft Position vs Playoff Success',
                        labels={'Avg_Draft_Position': 'Ø Draft Position', 'Playoff_Rate': 'Playoff Rate'},
                        template="plotly_dark"
                    )
                    fig2.update_layout(height=350)
                    st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.subheader("Manager-Player Loyalty")
            st.markdown("*Wer draftet immer wieder denselben Spieler?*")
            
            # Calculate loyalty
            loyalty_df = calculate_manager_player_loyalty(drafts_df, teams_df)
           
            if loyalty_df is not None and 'Years' in loyalty_df.columns:
                def fix_loyalty_years(x):
                    if pd.isna(x) or x == '' or str(x).lower() == 'nan':
                        return x
                    try:
                        years = str(x).split(', ')
                        fixed_years = []
                        for year in years:
                            year = year.strip()
                            if year and year.replace('.', '').replace('-', '').isdigit():
                                try:
                                    fixed_years.append(str(int(float(year))))
                                except (ValueError, TypeError):
                                    fixed_years.append(year)
                            elif year:
                                fixed_years.append(year)
                        return ', '.join(fixed_years) if fixed_years else str(x)
                    except Exception:
                        return str(x)
                
                loyalty_df['Years'] = loyalty_df['Years'].apply(fix_loyalty_years)
                
            if loyalty_df is not None and len(loyalty_df) > 0:
                # Top loyalty combinations
                st.markdown("### 💕 Stärkste Manager-Spieler Bindungen")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display top loyalty pairs
                    for i, (_, pair) in enumerate(loyalty_df.head(10).iterrows()):
                        st.markdown(f"""
                        <div class="loyalty-player">
                            <h4>#{i+1} {pair['Manager']} ❤️ {pair['Player']}</h4>
                            <p><strong>{pair['Times_Drafted']}x</strong> gedraftet in Jahren: {pair['Years']}</p>
                            <p>Ø Runde {pair['Avg_Draft_Round']} | Loyalty Score: {pair['Loyalty_Score']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Summary stats
                    st.markdown("### 📊 Loyalty Stats")
                    
                    # Most loyal manager
                    manager_loyalty = loyalty_df.groupby('Manager')['Times_Drafted'].sum().sort_values(ascending=False)
                    if len(manager_loyalty) > 0:
                        st.metric("Loyalster Manager", manager_loyalty.index[0], f"{manager_loyalty.iloc[0]} Total Drafts")
                    
                    # Most drafted player
                    player_popularity = loyalty_df.groupby('Player')['Times_Drafted'].sum().sort_values(ascending=False)
                    if len(player_popularity) > 0:
                        st.metric("Beliebtester Spieler", player_popularity.index[0], f"{player_popularity.iloc[0]}x gedraftet")
                    
                    # Average loyalty
                    avg_loyalty = loyalty_df['Times_Drafted'].mean()
                    st.metric("Ø Loyalty", f"{avg_loyalty:.1f} Drafts")
                
                # Full loyalty table
                st.markdown("### 📋 Vollständige Loyalty-Tabelle")
                
                # Prepare DataFrame for display - remove unwanted columns and round values
                display_loyalty_df = loyalty_df.copy()
                
                # Remove unwanted columns if they exist
                columns_to_remove = ['PlayerID', 'Unique_Seasons']
                for col in columns_to_remove:
                    if col in display_loyalty_df.columns:
                        display_loyalty_df = display_loyalty_df.drop(col, axis=1)
                
                # Round numerical columns to whole numbers
                numerical_columns = ['Avg_Draft_Round', 'Loyalty_Score', 'Avg_Draft_Position']
                for col in numerical_columns:
                    if col in display_loyalty_df.columns:
                        display_loyalty_df[col] = display_loyalty_df[col].round(0).astype(int)
                
                # Style the loyalty table
                def highlight_loyalty_score(val):
                    if pd.isna(val):
                        return ""
                    if val >= 15:
                        return "background-color: rgba(138, 43, 226, 0.4);"  # Purple for high loyalty
                    elif val >= 10:
                        return "background-color: rgba(138, 43, 226, 0.25);"
                    elif val >= 5:
                        return "background-color: rgba(138, 43, 226, 0.15);"
                    else:
                        return "background-color: rgba(138, 43, 226, 0.05);"
                
                styled_loyalty = display_loyalty_df.style.applymap(
                    highlight_loyalty_score, 
                    subset=['Loyalty_Score'] if 'Loyalty_Score' in display_loyalty_df.columns else []
                )
                
                st.dataframe(
                    styled_loyalty,
                    column_config={
                        "Manager": "Manager",
                        "Player": "Player",
                        "Times_Drafted": "Anzahl Drafts",
                        "Years": "Jahre",
                        "Avg_Draft_Round": "Ø Draft Runde",
                        "Loyalty_Score": "Loyalty Score"
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Loyalty visualization
                fig = px.treemap(
                    loyalty_df.head(20),
                    path=['Manager', 'Player'],
                    values='Times_Drafted',
                    color='Loyalty_Score',
                    color_continuous_scale='Viridis',
                    title='Manager-Player Loyalty Map (Top 20)',
                    template="plotly_dark"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("Keine Loyalty-Daten verfügbar. Diese Feature benötigt detaillierte Spieler-Draft-Daten.")
                st.markdown("""
                **Hinweis:** Für vollständige Player Analysis werden folgende Daten in deinem Google Sheet benötigt:
            
                - **mDrafts Sheet:** Spielernamen für jeden Draft Pick
                - **Team Rosters:** Welche Spieler in welchen Teams waren
                - **Season Results:** Verknüpfung von Spielern zu Championship/Finals Teams
            
                Sobald diese Daten verfügbar sind, werden hier automatisch echte Insights angezeigt!
                """)

    elif st.session_state.analysis_type == "📊 Categories":
        st.header("Statistik-Kategorien")
    
        # Überprüfe, ob die Daten geladen wurden
        if categories_df is not None and not categories_df.empty:
            
            # Filtere alle Zeilen heraus, die Turnovers = 0 haben
            filtered_categories_df = categories_df[categories_df['Turnovers'] != 0].copy()
            
            # Manager-Mapping korrekt implementieren
            if 'Manager' not in filtered_categories_df.columns:
                # Nutze das bestehende team_mapping aus create_team_mapping()
                # Format: {(TeamID, Year): FirstName}
                if 'team_mapping' in locals() and team_mapping:
                    # Erstelle Manager-Spalte basierend auf TeamID und Saison
                    def get_manager_name(row):
                        key = (row['TeamID'], row['Saison'])
                        return team_mapping.get(key, f"Team_{row['TeamID']}")
                    
                    filtered_categories_df['Manager'] = filtered_categories_df.apply(get_manager_name, axis=1)
                    groupby_column = 'Manager'
                else:
                    # Fallback auf TeamID, falls kein Mapping verfügbar
                    groupby_column = 'TeamID'
            else:
                groupby_column = 'Manager'
        
            # Tabs für die zwei Ansichten erstellen
            tab1, tab2 = st.tabs(["📈 Career Averages", "🥇 All-Time Stat Leaders"])
        
            # Statistische Berechnungen
            # Definiere die Spalten für Rohdaten und Prozentwerte mit Emojis
            raw_stats = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', '3PM', 'Turnovers']
            percentage_stats = ['FG%', 'FT%']
            stats_to_plot = raw_stats + percentage_stats
            
            # Emojis für jede Kategorie
            stat_emojis = {
                'Points': '🗑️',
                'Rebounds': '🙌',
                'Assists': '🤝✨',
                'Steals': '🥷',
                'Blocks': '☝🏽🚫',
                '3PM': '👌',
                'Turnovers': '🔄',
                'FG%': '🎯',
                'FT%': '💯'
            }
            
            # Farben für jede Kategorie definieren
            stat_colors = {
                'Points': '#FF6B6B',        # Rot
                'Rebounds': '#4B0082',      # Dunkelviolett
                'Assists': '#8B008B',       # Dunkelmagenta
                'Steals': '#00008B',        # Dunkelblau
                'Blocks': '#2F4F4F',        # Dunkelgrau
                '3PM': '#8B0000',           # Dunkelrot
                'Turnovers': '#008B8B',     # DunkelCyan
                'FG%': '#5C4033',           # Dunkelbraun
                'FT%': '#556B2F'            # Olivgruen
            }

            with tab1:
                st.subheader("Career Averages")
                st.markdown("Alle Stats gerechnet auf die Anzahl der gespielten Saisons. Dies gibt eine realistische Darstellung der Stärken/Schwächen. Manager, die schon länger dabei sind haben keinen Vorteil in der Auswertung.  Notiz: Die Statistiken für Saison 2014 sind nicht enthalten.")

                # Zähle die Anzahl der gespielten Jahre pro Manager/Team
                years_played = filtered_categories_df.groupby(groupby_column)['Saison'].nunique().rename("Years Played")

                # Berechne die Summen der Rohwerte
                raw_stats_sums = filtered_categories_df.groupby(groupby_column)[raw_stats].sum()

                # Berechne die Durchschnittswerte pro Jahr
                career_averages = raw_stats_sums.div(years_played, axis=0)

                # Füge die Prozentwerte hinzu (die Berechnung ist dieselbe wie im ersten Tab)
                percentage_averages = filtered_categories_df.groupby(groupby_column)[percentage_stats].mean()
                career_averages = pd.concat([career_averages, percentage_averages], axis=1)
                
                # Runde die Rohwerte auf ganze Zahlen, lasse Prozentwerte als Dezimalzahlen
                for stat in raw_stats:
                    career_averages[stat] = career_averages[stat].round(0).astype(int)

                # Erstelle für jede Kategorie ein horizontales Balkendiagramm für die Top 10
                for stat in stats_to_plot:
                    # Turnovers wird aufsteigend (klein nach groß) sortiert, alle anderen absteigend
                    ascending_sort = True if stat == 'Turnovers' else False
                    sorted_stats = career_averages.sort_values(by=stat, ascending=ascending_sort).head(10)
                    
                    # Spezielle Behandlung für Turnovers: Minimum 5 Saisons
                    if stat == 'Turnovers':
                        # Filtere Manager mit mindestens 5 gespielten Saisons
                        qualified_managers = years_played[years_played >= 5].index
                        filtered_stats = career_averages.loc[qualified_managers]
                        sorted_stats = filtered_stats.sort_values(by=stat, ascending=ascending_sort).head(10)
                    else:
                        sorted_stats = career_averages.sort_values(by=stat, ascending=ascending_sort).head(10)
                    
                    # Formatiere Prozentwerte für die Anzeige
                    if stat in percentage_stats:
                        display_stats = sorted_stats.copy()
                        display_stats[stat] = (display_stats[stat] * 100).round(1)
                        x_format = '.1f'
                        x_suffix = '%'
                        text_values = [f"{val:.1f}%" for val in display_stats[stat]]
                    else:
                        display_stats = sorted_stats
                        x_format = 'd'
                        x_suffix = ''
                        text_values = [f"{int(val)}" for val in display_stats[stat]]
                
                    title = f"Top 10 - Career Average {stat_emojis[stat]} {stat}"

                    fig = px.bar(
                        display_stats,
                        y=display_stats.index,
                        x=stat,
                        orientation='h',
                        title=title,
                        color_discrete_sequence=[stat_colors[stat]],
                        text=text_values,
                        template="plotly_dark"
                    )
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending' if not ascending_sort else 'total descending'},
                        xaxis_title=f"{stat_emojis[stat]} {stat}{x_suffix}",
                        yaxis_title="Manager" if groupby_column == 'Manager' else "Team Name"
                    )
                    
                    # Formatiere x-Achsen Labels für Prozentwerte
                    if stat in percentage_stats:
                        fig.update_xaxes(tickformat='.1f', ticksuffix='%')
                    
                    # Text innerhalb der Balken positionieren
                    fig.update_traces(textposition='inside', textfont_size=12, textfont_color='white')
                    
                    st.plotly_chart(fig, use_container_width=True)

                # --- Dynamische Tabelle für alle Manager ---
                st.subheader("Vollständige Tabelle aller Manager")
                
                # Dropdown-Menü, um die Kategorie auszuwählen
                selected_category = st.selectbox(
                    "Wählen Sie eine Kategorie:",
                    options=stats_to_plot,
                    key="tab1_selectbox"
                )
            
                # Sortiere die vollständige Tabelle basierend auf der ausgewählten Kategorie
                ascending_sort = True if selected_category == 'Turnovers' else False
                filtered_table = career_averages.sort_values(by=selected_category, ascending=ascending_sort)
                
                # Erstelle eine Kopie für die Anzeige mit formatierten Prozentwerten
                display_table = filtered_table.copy()
                for stat in percentage_stats:
                    if stat in display_table.columns:
                        display_table[stat] = (display_table[stat] * 100).round(1).astype(str) + '%'

                # Zeige die gefilterte Tabelle an
                st.dataframe(display_table, use_container_width=True)

            with tab2:
                st.subheader("All-Time Stat Leaders")
                st.markdown("Alle Statistiken seit Anbeginn der Domination League zu einer Summe addiert. Manager, die schon länger dabei sind haben logischerweise einen Vorteil in der Auswertung, da sie mehr Jahre hatten, um Stats zu sammeln.  Notiz: Die Statistiken für Saison 2014 sind nicht enthalten.")

                agg_funcs = {stat: 'sum' for stat in raw_stats}
                agg_funcs.update({stat: 'mean' for stat in percentage_stats})
            
                all_time_stats = filtered_categories_df.groupby(groupby_column).agg(agg_funcs)

                # Erstelle für jede Kategorie ein horizontales Balkendiagramm für die Top 10
                for stat in stats_to_plot:
                    # Turnovers wird aufsteigend (klein nach groß) sortiert, alle anderen absteigend
                    ascending_sort = True if stat == 'Turnovers' else False
                    sorted_stats = all_time_stats.sort_values(by=stat, ascending=ascending_sort).head(10)
                   
                    # Spezielle Behandlung für Turnovers: Minimum 5 Saisons
                    if stat == 'Turnovers':
                        # Filtere Manager mit mindestens 5 gespielten Saisons
                        qualified_managers = years_played[years_played >= 5].index
                        filtered_stats = all_time_stats.loc[qualified_managers]
                        sorted_stats = filtered_stats.sort_values(by=stat, ascending=ascending_sort).head(10)
                    else:
                        sorted_stats = all_time_stats.sort_values(by=stat, ascending=ascending_sort).head(10)                    
                  
                    # Formatiere Prozentwerte für die Anzeige
                    if stat in percentage_stats:
                        display_stats = sorted_stats.copy()
                        display_stats[stat] = (display_stats[stat] * 100).round(1)
                        x_format = '.1f'
                        x_suffix = '%'
                        text_values = [f"{val:.1f}%" for val in display_stats[stat]]
                    else:
                        display_stats = sorted_stats
                        x_format = 'd'
                        x_suffix = ''
                        text_values = [f"{int(val)}" for val in display_stats[stat]]
                
                    title = f"Top 10 - All-Time {stat_emojis[stat]} {stat}"

                    fig = px.bar(
                        display_stats,
                        y=display_stats.index,
                        x=stat,
                        orientation='h',
                        title=title,
                        color_discrete_sequence=[stat_colors[stat]],
                        text=text_values,
                        template="plotly_dark"
                    )
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending' if not ascending_sort else 'total descending'},
                        xaxis_title=f"{stat_emojis[stat]} {stat}{x_suffix}",
                        yaxis_title="Manager" if groupby_column == 'Manager' else "Team Name"
                    )
                    
                    # Formatiere x-Achsen Labels für Prozentwerte
                    if stat in percentage_stats:
                        fig.update_xaxes(tickformat='.1f', ticksuffix='%')
                    
                    # Text innerhalb der Balken positionieren
                    fig.update_traces(textposition='inside', textfont_size=12, textfont_color='white')
                    
                    st.plotly_chart(fig, use_container_width=True)

                # --- Dynamische Tabelle für alle Manager ---
                st.subheader("Vollständige Tabelle aller Manager")
                
                # Dropdown-Menü, um die Kategorie auszuwählen
                selected_category = st.selectbox(
                    "Wählen Sie eine Kategorie:",
                    options=stats_to_plot,
                    key="tab2_selectbox"
                )
            
                # Sortiere die vollständige Tabelle basierend auf der ausgewählten Kategorie
                ascending_sort = True if selected_category == 'Turnovers' else False
                filtered_table = all_time_stats.sort_values(by=selected_category, ascending=ascending_sort)
                
                # Erstelle eine Kopie für die Anzeige mit formatierten Prozentwerten
                display_table = filtered_table.copy()
                for stat in percentage_stats:
                    if stat in display_table.columns:
                        display_table[stat] = (display_table[stat] * 100).round(1).astype(str) + '%'

                # Zeige die gefilterte Tabelle an
                st.dataframe(display_table, use_container_width=True)

        else:
            st.warning("Die Daten für 'Categories' konnten nicht geladen werden.")

    elif st.session_state.analysis_type == "🤝 Trades":
        st.title("🤝 Trade-Übersicht")
        
        # Laden der Trade-Daten
        trades_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUsvt5i3VEhZkg_bC_fGzJSg_xjkEsQVvkZ9D7uyY-d9-ExS5pTZUYpR9qCkIin1ZboCh4o6QcCBe3/pub?gid=58770562&single=true&output=csv"
    
        try:
            trades_df = pd.read_csv(trades_url)
        except Exception as e:
            st.error(f"❌ Fehler beim Laden der Trade-Daten: {str(e)}")
            st.stop()
        
        # Saison-Filter
        seasons = sorted(trades_df['Saison'].unique(), reverse=True)
        selected_season = st.selectbox("Saison auswählen:", ['Alle Saisons'] + list(seasons))
        
        # Daten filtern
        if selected_season != 'Alle Saisons':
            filtered_trades = trades_df[trades_df['Saison'] == selected_season]
        else:
            filtered_trades = trades_df
        
        # Grundlegende Statistiken
        col1, col2, col3 = st.columns(3)
        with col1:
            total_trades = len(filtered_trades['TradeID'].unique())
            st.metric("Gesamte Trades", total_trades)
        
        with col2:
            if not filtered_trades.empty:
                trades_per_season = filtered_trades.groupby('Saison')['TradeID'].nunique().mean()
                st.metric("Ø Trades pro Saison", f"{trades_per_season:.1f}")
        
        with col3:
            if not filtered_trades.empty:
                players_traded = len(filtered_trades['playerName'].unique())
                st.metric("Verschiedene Spieler", players_traded)
        
        # Trade-Details Tabelle
        st.subheader("📋 Trade-Details")
        
        if not filtered_trades.empty:
            # Gruppierung nach TradeID für bessere Übersicht
            trade_summary = []
            
            for trade_id in filtered_trades['TradeID'].unique():
                trade_data = filtered_trades[filtered_trades['TradeID'] == trade_id]
                
                # Teams und Spieler nach Teams gruppiert sammeln
                teams = set()
                team_players = {}
                saison = trade_data['Saison'].iloc[0]
                
                for _, row in trade_data.iterrows():
                    from_team = row['Team1']
                    to_team = row['Team2']
                    player = row['playerName']
                    
                    teams.add(from_team)
                    teams.add(to_team)
                    
                    # Spieler dem abgebenden Team zuordnen
                    if from_team not in team_players:
                        team_players[from_team] = []
                    team_players[from_team].append(player)
                
                teams_list = list(teams)
                
                # Bessere Darstellung der getauschten Spieler
                if len(teams_list) == 2:
                    team1_players = team_players.get(teams_list[0], [])
                    team2_players = team_players.get(teams_list[1], [])
                    
                    # Formatierung: Team1 Spieler ↔ Team2 Spieler
                    team1_str = ", ".join(team1_players) if team1_players else ""
                    team2_str = ", ".join(team2_players) if team2_players else ""
                    
                    if team1_str and team2_str:
                        traded_players = f"{team1_str} 🔁 {team2_str}"
                    elif team1_str:
                        traded_players = f"{team1_str} → {teams_list[1]}"
                    elif team2_str:
                        traded_players = f"{teams_list[0]} → {team2_str}"
                    else:
                        traded_players = "Keine Spielerdaten"
                else:
                    # Fallback für komplexere Trades
                    all_players = []
                    for team, players in team_players.items():
                        all_players.extend(players)
                    traded_players = "\n".join(all_players)
                
                trade_summary.append({
                    'Saison': saison,
                    'Beteiligte Teams': f"{teams_list[0]} 🤝 {teams_list[1]}" if len(teams_list) == 2 else ", ".join(teams_list),
                    'Getauschte Spieler': traded_players,
                    'Anzahl Spieler': len([p for players in team_players.values() for p in players]),
                    'TradeID': trade_id  # Versteckt für interne Verwendung
                })
            
            # DataFrame für die Anzeige
            summary_df = pd.DataFrame(trade_summary)
            summary_df = summary_df.sort_values(['Saison', 'TradeID'], ascending=[False, False])
            
            # Interaktive Tabelle (ohne TradeID)
            display_df = summary_df.drop('TradeID', axis=1)
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Zusätzliche Analysen
            st.subheader("📊 Trade-Aktivität")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Trades pro Saison
                trades_by_season = filtered_trades.groupby('Saison')['TradeID'].nunique().reset_index()
                trades_by_season.columns = ['Saison', 'Anzahl Trades']
                
                st.write("**Trades pro Saison:**")
                chart_data = trades_by_season.set_index('Saison')
                st.bar_chart(chart_data)
            
            with col2:
                # Aktivste Teams - korrekte Zählung pro unique Trade
                team_trade_counts = {}
                
                for trade_id in filtered_trades['TradeID'].unique():
                    trade_data = filtered_trades[filtered_trades['TradeID'] == trade_id]
                    involved_teams = set()
                    
                    for _, row in trade_data.iterrows():
                        involved_teams.add(row['Team1'])
                        involved_teams.add(row['Team2'])
                    
                    # Jedes beteiligte Team bekommt +1 für diesen Trade
                    for team in involved_teams:
                        if team not in team_trade_counts:
                            team_trade_counts[team] = 0
                        team_trade_counts[team] += 1
                
                # Sortierung nach Anzahl Trades (descending) und als DataFrame für bessere Kontrolle
                team_counts_sorted = sorted(team_trade_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                
                # DataFrame für die Anzeige erstellen
                team_chart_df = pd.DataFrame(team_counts_sorted, columns=['Team', 'Anzahl Trades'])
                team_chart_df = team_chart_df.set_index('Team')
                
                st.write("**Aktivste Teams (Top 10):**")
                st.bar_chart(team_chart_df['Anzahl Trades'])
            
            # Detailansicht einzelner Trades
            st.subheader("🔍 Trade-Details")
            
            # Trade-Optionen mit benutzerfreundlichen Namen
            trade_options = ['Bitte wählen...']
            trade_mapping = {}
            
            for _, row in summary_df.iterrows():
                display_name = f"{row['Saison']} - {row['Beteiligte Teams']}"
                trade_options.append(display_name)
                trade_mapping[display_name] = row['TradeID']
            
            selected_trade_display = st.selectbox(
                "Trade auswählen für Details:",
                trade_options
            )
            
            if selected_trade_display != 'Bitte wählen...':
                selected_trade = trade_mapping[selected_trade_display]
                trade_details = filtered_trades[filtered_trades['TradeID'] == selected_trade]
                
                st.write(f"**Saison:** {trade_details['Saison'].iloc[0]}")
                st.write(f"**Beteiligte Teams:** {selected_trade_display.split(' - ')[1]}")
                
                # Spieler nach Teams gruppiert anzeigen
                teams_in_trade = {}
                for _, row in trade_details.iterrows():
                    from_team = row['Team1']
                    to_team = row['Team2']
                    player = row['playerName']
                    
                    if from_team not in teams_in_trade:
                        teams_in_trade[from_team] = {'gibt_ab': [], 'bekommt': []}
                    if to_team not in teams_in_trade:
                        teams_in_trade[to_team] = {'gibt_ab': [], 'bekommt': []}
                    
                    teams_in_trade[from_team]['gibt_ab'].append(player)
                    teams_in_trade[to_team]['bekommt'].append(player)
                
                for team, actions in teams_in_trade.items():
                    if actions['gibt_ab'] or actions['bekommt']:
                        st.write(f"**{team}:**")
                        if actions['gibt_ab']:
                            st.write(f"  • Gibt ab: {', '.join(actions['gibt_ab'])}")
                        if actions['bekommt']:
                            st.write(f"  • Bekommt: {', '.join(actions['bekommt'])}")
        
        else:
            st.info("Keine Trades für die ausgewählte Saison gefunden.")

    # Homepage Dashboard - fügen Sie das am ENDE Ihrer main() function ein, nach allen elif statements
    else:  # Default homepage when no analysis_type is selected
        # Liga Header mit Logo
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Logo anzeigen (Google Drive direkte URL)
            logo_url = "https://drive.google.com/uc?export=view&id=1uYESKqDX62PrZzZoJlmoMdt0mgobSWXu"
            try:
                st.image(logo_url, width=300)
            except:
                st.markdown("### 🏀 Fantasy Basketball Liga")
        
        st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1>Welcome to Domination Analytics</h1>
            <p style='font-size: 1.2rem; color: #888;'>Seasons 2014-2025 • 11 Jahre Liga-Geschichte</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Top 3 Siegertreppchen vom aktuellen Jahr
        st.markdown("### 🏆 Top 3 des letzten Jahres")
        
        try:
            # Ermittlung der Top 3 vom aktuellen Jahr (2025) aus seasons_df
            latest_champions = seasons_df[seasons_df['Season'] == 2025].nlargest(3, 'Points')
            
            if len(latest_champions) >= 3:
                # Siegertreppchen Layout
                col1, col2, col3 = st.columns([1, 2, 1])
                
                # 2. Platz (links)
                with col1:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #C0C0C0, #A8A8A8); 
                               border-radius: 15px; margin-top: 30px;'>
                        <h3 style='margin: 0; color: #333;'>🥈</h3>
                        <h4 style='margin: 5px 0; color: #333;'>{latest_champions.iloc[1]['Manager']}</h4>
                        <p style='margin: 0; font-weight: bold; color: #555;'>{latest_champions.iloc[1]['Points']:.1f} Pts</p>
                        <p style='margin: 0; font-size: 0.9rem; color: #666;'>2nd Place</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 1. Platz (mitte, höher)
                with col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 25px; background: linear-gradient(135deg, #FFD700, #FFA500); 
                               border-radius: 15px; box-shadow: 0 4px 8px rgba(255,215,0,0.3);'>
                        <h2 style='margin: 0; color: #333;'>👑</h2>
                        <h3 style='margin: 10px 0; color: #333;'>{latest_champions.iloc[0]['Manager']}</h3>
                        <p style='margin: 0; font-weight: bold; font-size: 1.2rem; color: #333;'>{latest_champions.iloc[0]['Points']:.1f} Pts</p>
                        <p style='margin: 0; font-size: 1rem; color: #555;'>🏆 CHAMPION 2025</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # 3. Platz (rechts)
                with col3:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #CD7F32, #8B4513); 
                               border-radius: 15px; margin-top: 30px;'>
                        <h3 style='margin: 0; color: #FFF;'>🥉</h3>
                        <h4 style='margin: 5px 0; color: #FFF;'>{latest_champions.iloc[2]['Manager']}</h4>
                        <p style='margin: 0; font-weight: bold; color: #FFF;'>{latest_champions.iloc[2]['Points']:.1f} Pts</p>
                        <p style='margin: 0; font-size: 0.9rem; color: #DDD;'>3rd Place</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Keine ausreichenden Daten für 2025 Top 3 verfügbar.")
        except Exception as e:
            st.info("Top 3 Daten werden geladen...")
        
        st.markdown("---")
        
        # Navigation Cards
        st.markdown("### 📍 Navigation")
        st.markdown("*Wähle einen Bereich für detaillierte Analysen:*")
        
        # Row 1: Team & Manager Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("⛹🏽‍♂️ Team-View", use_container_width=True, key="nav_team"):
                st.session_state.analysis_type = "⛹🏽‍♂️ Team-View"
                st.rerun()
            st.markdown("""
            <div style='padding: 15px; background: #f0c7c7; border-radius: 10px; margin-top: -10px;'>
                <p style='margin: 0; color: #888; font-size: 0.9rem;'>
                    📊 Team Rankings, Saisonverläufe und Performance-Trends aller Manager
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🥊 Head-to-Head", use_container_width=True, key="nav_h2h"):
                st.session_state.analysis_type = "🥊 Head-to-Head"
                st.rerun()
            st.markdown("""
            <div style='padding: 15px; background: #1E1E1E; border-radius: 10px; margin-top: -10px;'>
                <p style='margin: 0; color: #888; font-size: 0.9rem;'>
                    ⚔️ Direkte Vergleiche zwischen Managern - Wer dominiert wen?
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 2: Championships & Performance  
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏆 Playoff Performance", use_container_width=True, key="nav_playoffs"):
                st.session_state.analysis_type = "🏆 Playoff Performance"
                st.rerun()
            st.markdown("""
            <div style='padding: 15px; background: #1E1E1E; border-radius: 10px; margin-top: -10px;'>
                <p style='margin: 0; color: #888; font-size: 0.9rem;'>
                    🏆 Championship-Geschichte, Finals und Playoff-Erfolge
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🏅 Medal Overview", use_container_width=True, key="nav_medals"):
                st.session_state.analysis_type = "🏅 Medal Overview"
                st.rerun()
            st.markdown("""
            <div style='padding: 15px; background: #1E1E1E; border-radius: 10px; margin-top: -10px;'>
                <p style='margin: 0; color: #888; font-size: 0.9rem;'>
                    🥇 Alle Titel und Auszeichnungen - Wer sammelt die meisten Medals?
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 3: Draft & Player Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🎯 Drafts", use_container_width=True, key="nav_drafts"):
                st.session_state.analysis_type = "🎯 Drafts"
                st.rerun()
            st.markdown("""
            <div style='padding: 15px; background: #1E1E1E; border-radius: 10px; margin-top: -10px;'>
                <p style='margin: 0; color: #888; font-size: 0.9rem;'>
                    🎯 Draft-Strategien, Pick-Analysen und Erfolgsraten
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("👨‍💼 Player Analysis", use_container_width=True, key="nav_players"):
                st.session_state.analysis_type = "👨‍💼 Player Analysis"
                st.rerun()
            st.markdown("""
            <div style='padding: 15px; background: #1E1E1E; border-radius: 10px; margin-top: -10px;'>
                <p style='margin: 0; color: #888; font-size: 0.9rem;'>
                    👑 Championship-DNA, Legenden und Manager-Spieler Loyalität
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 4: Categories & Trades
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Categories", use_container_width=True, key="nav_categories"):
                st.session_state.analysis_type = "📊 Categories"
                st.rerun()
            st.markdown("""
            <div style='padding: 15px; background: #1E1E1E; border-radius: 10px; margin-top: -10px;'>
                <p style='margin: 0; color: #888; font-size: 0.9rem;'>
                    📈 Kategorie-Performance - Stärken und Schwächen analysieren
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🤝 Trades", use_container_width=True, key="nav_trades"):
                st.session_state.analysis_type = "🤝 Trades"
                st.rerun()
            st.markdown("""
            <div style='padding: 15px; background: #1E1E1E; border-radius: 10px; margin-top: -10px;'>
                <p style='margin: 0; color: #888; font-size: 0.9rem;'>
                    🤝 Trade-Historie, aktivste Manager und Deal-Analysen
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Liga-Info Footer
        st.markdown("""
        <div style='text-align: center; padding: 20px; color: #666;'>
            <p>🏀 <strong>Fantasy Basketball Liga</strong> • Seit 2015 • 11 Saisons</p>
            <p style='font-size: 0.9rem;'>Daten werden live aus Google Sheets geladen</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

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
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #FF6B35;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .medal-container {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }
    .favorite-opponent {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .nightmare-opponent {
        background-color: #fee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        margin: 0.5rem 0;
    }
    .champion-player {
        background-color: #fff8dc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FFD700;
        margin: 0.5rem 0;
    }
    .legend-player {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4169E1;
        margin: 0.5rem 0;
    }
    .loyalty-player {
        background-color: #f5f0ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #8A2BE2;
        margin: 0.5rem 0;
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
        
        teams_df = pd.read_csv(teams_url)
        matchups_df = pd.read_csv(matchups_url)
        drafts_df = pd.read_csv(drafts_url)
        categories_df = pd.read_csv(categories_url)
        seasons_df = pd.read_csv(seasons_url)
        
        return teams_df, matchups_df, drafts_df, categories_df, seasons_df
    
    except (URLError, HTTPError) as e:
        st.error(f"Fehler beim Laden der Daten: {e.reason}")
        return None, None, None, None

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
            'Over_Under': 'Over/Under Score'
        },
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
                    'Made_Playoffs': final_rank <= 6,  # Assuming top 6 make playoffs
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
    for _, player_row in champ_counts.iterrows():
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
    for _, player_row in finals_counts.iterrows():
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
    
    return champ_df, finals_df

def calculate_legend_analysis(drafts_df, teams_df):
    """Calculate legend analysis - first round superstars and playoff heroes"""
    
    if drafts_df is None or drafts_df.empty:
        st.info("Keine Spielerdaten für Legend Analysis verfügbar.")
        return None, None
    
    # First Round Superstars - directly from Round column
    first_round_drafts = drafts_df[drafts_df['Round'] == 1]  # Changed this line
    
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
    
    # Keep the playoff heroes part as is, or simplify it similarly
    playoff_heroes_df = None  # Simplified for now
    
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
    
    # Basis-Aggregation
    agg_dict = {
        season_col: [
            'count',  # Wie oft gedraftet
            'nunique',  # In wie vielen verschiedenen Seasons
            lambda x: ', '.join(map(str, sorted(x.unique())))  # Welche Jahre
        ]
    }
    
    # Füge Draft Position hinzu wenn verfügbar
    if draft_pos_col:
        agg_dict[draft_pos_col] = 'mean'
    
    # Füge Round hinzu wenn verfügbar
    if round_col:
        agg_dict[round_col] = 'mean'
    
    # Berechne Loyalty-Kombinationen
    loyalty_combinations = drafts_df.groupby(['TeamID', 'PlayerID']).agg(agg_dict).round(1)
    
    # Flatten column names
    new_columns = ['Times_Drafted', 'Unique_Seasons', 'Years']
    
    if draft_pos_col:
        new_columns.append('Avg_Draft_Position')
    if round_col:
        new_columns.append('Avg_Draft_Round')
    
    loyalty_combinations.columns = new_columns
    loyalty_combinations = loyalty_combinations.reset_index()
    
    # Mappe TeamID zu Manager Namen (jahr-spezifisch)
    if teams_df is not None and not teams_df.empty:
        
        # Erstelle jahr-spezifisches Team-Mapping
        manager_col = None
        if 'First Name' in teams_df.columns:
            manager_col = 'First Name'
        elif 'Manager' in teams_df.columns:
            manager_col = 'Manager'
        elif 'Name' in teams_df.columns:
            manager_col = 'Name'
        else:
            st.error(f"Keine Manager-Name Spalte in teams_df gefunden. Verfügbare Spalten: {teams_df.columns.tolist()}")
            return None
        
        # Merge drafts_df mit teams_df basierend auf TeamID UND Season/Year
        teams_season_col = None
        for col in ['Season', 'Year', 'season', 'year']:
            if col in teams_df.columns:
                teams_season_col = col
                break
        
        if teams_season_col is None:
            st.error("Keine Season/Year Spalte in teams_df gefunden für jahr-spezifisches Mapping!")
            return None
                
        # Merge um Manager-Namen für jede Draft-Zeile zu bekommen
        drafts_with_managers = drafts_df.merge(
            teams_df[['TeamID', teams_season_col, manager_col]], 
            left_on=['TeamID', season_col], 
            right_on=['TeamID', teams_season_col], 
            how='left'
        )
        
        # Überprüfe auf fehlende Manager-Zuordnungen
        missing_managers = drafts_with_managers[manager_col].isna().sum()
        if missing_managers > 0:
            st.warning(f"DEBUG - {missing_managers} Drafts konnten keinem Manager zugeordnet werden")
            st.write("Beispiel-Zeilen ohne Manager:")
            st.dataframe(drafts_with_managers[drafts_with_managers[manager_col].isna()][['TeamID', season_col, 'PlayerName']].head())
        
        # Jetzt gruppieren wir nach dem echten Manager-Namen und PlayerID
        agg_dict_new = {
            season_col: [
                'count',  # Wie oft gedraftet
                'nunique',  # In wie vielen verschiedenen Seasons
                lambda x: ', '.join(map(str, sorted(x.unique())))  # Welche Jahre
            ]
        }
        
        # Füge Draft Position hinzu wenn verfügbar
        if draft_pos_col:
            agg_dict_new[draft_pos_col] = 'mean'
        
        # Füge Round hinzu wenn verfügbar
        if round_col:
            agg_dict_new[round_col] = 'mean'
        
        # Neue Gruppierung nach echtem Manager-Namen
        loyalty_combinations = drafts_with_managers.groupby([manager_col, 'PlayerID']).agg(agg_dict_new).round(1)
        
        # Flatten column names
        new_columns = ['Times_Drafted', 'Unique_Seasons', 'Years']
        
        if draft_pos_col:
            new_columns.append('Avg_Draft_Position')
        if round_col:
            new_columns.append('Avg_Draft_Round')
        
        loyalty_combinations.columns = new_columns
        loyalty_combinations = loyalty_combinations.reset_index()
        
        # Benenne die Manager-Spalte um
        loyalty_combinations = loyalty_combinations.rename(columns={manager_col: 'Manager'})
        
    else:
        # Fallback wenn kein teams_df vorhanden
        loyalty_combinations = drafts_df.groupby(['TeamID', 'PlayerID']).agg(agg_dict).round(1)
        new_columns = ['Times_Drafted', 'Unique_Seasons', 'Years']
        if draft_pos_col:
            new_columns.append('Avg_Draft_Position')
        if round_col:
            new_columns.append('Avg_Draft_Round')
        loyalty_combinations.columns = new_columns
        loyalty_combinations = loyalty_combinations.reset_index()
        loyalty_combinations['Manager'] = loyalty_combinations['TeamID']  # Fallback
    
    # Mappe PlayerID zu Spieler Namen aus drafts_df
    # Erstelle Player-Mapping aus den ersten Vorkommen jeder PlayerID
    player_mapping = drafts_df.groupby('PlayerID')['PlayerName'].first().to_dict()
    loyalty_combinations['Player'] = loyalty_combinations['PlayerID'].map(player_mapping)
    
    # Calculate loyalty score
    loyalty_score = loyalty_combinations['Times_Drafted'] * 3 + loyalty_combinations['Unique_Seasons'] * 2
    
    # Bonus für frühe Runden (nur wenn Round-Spalte verfügbar)
    if 'Avg_Draft_Round' in loyalty_combinations.columns:
        max_round = drafts_df[round_col].max() if round_col else 14
        round_bonus = (max_round + 1 - loyalty_combinations['Avg_Draft_Round']) * 0.5
        loyalty_score += round_bonus
    
    loyalty_combinations['Loyalty_Score'] = loyalty_score.round(2)
    
    # Sort by loyalty score (höchste zuerst)
    loyalty_combinations = loyalty_combinations.sort_values('Loyalty_Score', ascending=False)
    
    # Filter: nur Manager-Spieler mit mehr als 1 Draft
    loyalty_combinations = loyalty_combinations[loyalty_combinations['Times_Drafted'] > 1]
    
    # Bereinige NaN/None Werte für Treemap
    loyalty_combinations = loyalty_combinations.dropna(subset=['Manager', 'Player'])
    
    return loyalty_combinations
    
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

# Main app
def main():
    st.markdown('<h1 class="main-header">🏀 Fantasy Basketball Analytics</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        teams_df, matchups_df, drafts_df, categories_df, seasons_df = load_data()
    
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
    
    if processed_df is None:
        st.error("Error processing matchup data.")
        return
    
    # Sidebar with button navigation
    st.sidebar.title("Navigation")
    
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
    
    # Initialize session state if not exists
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = "👥 Team-View"
    
    analysis_type = st.session_state.analysis_type
    
    # Main content based on selection
    if analysis_type == "👥 Team-View":
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
                    
                    if available_columns:
                        # Erstelle Display-Tabelle mit verfügbaren Spalten
                        display_table = manager_data[available_columns].copy()
                        
                        # Sortiere nach Saison absteigend (neueste zuerst)
                        if 'Saison' in display_table.columns:
                            display_table = display_table.sort_values('Saison', ascending=False)
                        
                        # Formatiere Win-Percentage als Prozentwert falls vorhanden
                        if 'Win-Percentage %' in display_table.columns:
                            # Falls die Werte als Dezimalzahlen vorliegen (0.75 statt 75)
                            if display_table['Win-Percentage %'].max() <= 1:
                                display_table['Win-Percentage %'] = (display_table['Win-Percentage %'] * 100).round(1)
                            display_table['Win-Percentage %'] = display_table['Win-Percentage %'].astype(str) + '%'
                        
                        # Zeige die Tabelle an
                        st.dataframe(display_table, use_container_width=True, hide_index=True)
                        
                        # Zusätzliche Statistiken
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
                    
                    else:
                        st.warning("Die benötigten Spalten wurden im Datensatz nicht gefunden.")
                        st.info("Verfügbare Spalten: " + ", ".join(manager_data.columns.tolist()))
                
                else:
                    st.warning(f"Keine Daten für Manager '{selected_manager}' gefunden.")
        
        else:
            st.warning("Die Seasons-Daten konnten nicht geladen werden.")
            
    elif analysis_type == "🥊 Head-to-Head":
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

    elif analysis_type == "🏆 Playoff Performance":
        st.header("Playoff Performance Analysis")
        
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
        
    elif analysis_type == "🏅 Medal Overview":
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
                if all(col in seasons_df.columns for col in ['First Name', 'Wins', 'Losses', 'Ties', 'Year']):
                    # Group by First Name and calculate statistics
                    eternal_stats = seasons_df.groupby('First Name').agg({
                        'Wins': 'sum',
                        'Losses': 'sum', 
                        'Ties': 'sum',
                        'Year': 'count'  # Count of seasons played
                    }).reset_index()
                    
                    # Rename columns for consistency
                    eternal_stats = eternal_stats.rename(columns={
                        'First Name': 'Manager',
                        'Year': 'Gespielte Saisons'
                    })
                
                # Rename the Season count column
                eternal_stats = eternal_stats.rename(columns={'Season': 'Gespielte Saisons'})
                
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
                st.error("seasons_df ist nicht verfügbar. Bitte stellen Sie sicher, dass die Daten geladen wurden.")
    
    elif analysis_type == "🎯 Drafts":
        st.header("Draft Analysis")
        
        # Process draft data
        draft_analysis_df = process_draft_data(drafts_df, teams_df)
        
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
                    color_continuous_scale='RdYlGn'
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
                        color_continuous_scale='RdYlGn'
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

    elif analysis_type == "👨‍💼 Player Analysis":
        st.header("Player Analysis")
        
        # Tabs for different player analyses
        tab1, tab2, tab3 = st.tabs(["🏆 Championship DNA", "🎯 Legenden", "📊 Manager-Player Loyalty"])
        
        with tab1:
            st.subheader("Championship DNA")
            st.markdown("*Welche Spieler standen am häufigsten in Championship-Teams?*")
            
            # Calculate championship DNA
            champ_df, finals_df = calculate_championship_dna(drafts_df, teams_df)
            
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
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Legenden")
            
            # Calculate legend analysis
            first_round_df, playoff_heroes_df = calculate_legend_analysis(drafts_df, teams_df)
            
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
                        color_continuous_scale='Blues_r'
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
                            <p><strong>{player['Playoff_Appearances']}</strong> Playoff Appearances</p>
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
                        labels={'Avg_Draft_Position': 'Ø Draft Position', 'Playoff_Rate': 'Playoff Rate'}
                    )
                    fig2.update_layout(height=350)
                    st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            st.subheader("Manager-Player Loyalty")
            st.markdown("*Wer draftet immer wieder denselben Spieler?*")
            
            # Calculate loyalty
            loyalty_df = calculate_manager_player_loyalty(drafts_df, teams_df)
            
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
                
                styled_loyalty = loyalty_df.style.applymap(
                    highlight_loyalty_score, 
                    subset=['Loyalty_Score']
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
                    title='Manager-Player Loyalty Map (Top 20)'
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

    elif analysis_type == "📊 Categories":
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
                        text=text_values
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
                        text=text_values
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
            
if __name__ == "__main__":
    main()

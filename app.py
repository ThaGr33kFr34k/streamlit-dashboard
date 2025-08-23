import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import random

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
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    """Load data from Google Sheets"""
    try:
        # Replace with your actual Google Sheets URLs (CSV export format)
        # Format: https://docs.google.com/spreadsheets/d/SHEET_ID/export?format=csv&gid=TAB_ID
        
        # You'll need to replace these URLs with your actual sheet URLs
        teams_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUsvt5i3VEhZkg_bC_fGzJSg_xjkEsQVvkZ9D7uyY-d9-ExS5pTZUYpR9qCkIin1ZboCh4o6QcCBe3/pub?gid=648434164&single=true&output=csv"
        matchups_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTUsvt5i3VEhZkg_bC_fGzJSg_xjkEsQVvkZ9D7uyY-d9-ExS5pTZUYpR9qCkIin1ZboCh4o6QcCBe3/pub?gid=652199133&single=true&output=csv"
        
        # Load the data
        teams_df = pd.read_csv(teams_url)
        matchups_df = pd.read_csv(matchups_url)
        
        return teams_df, matchups_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

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
        teams_df, matchups_df = load_data()
    
    if teams_df is None or matchups_df is None:
        st.error("Please update the Google Sheets URLs in the code with your actual sheet URLs.")
        st.info("""
        To get your Google Sheets CSV URLs:
        1. Open your Google Sheet
        2. Go to File → Share → Publish to web
        3. Select the tab (mTeams or mMatchups)
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
    if st.sidebar.button("🥊 Head-to-Head", use_container_width=True):
        st.session_state.analysis_type = "🥊 Head-to-Head"
    if st.sidebar.button("🏆 Playoff Performance", use_container_width=True):
        st.session_state.analysis_type = "🏆 Playoff Performance"
    if st.sidebar.button("🏅 Medal Overview", use_container_width=True):
        st.session_state.analysis_type = "🏅 Medal Overview"
    
    # Initialize session state if not exists
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = "🥊 Head-to-Head"
    
    analysis_type = st.session_state.analysis_type
    
    # Main content based on selection
    if analysis_type == "🥊 Head-to-Head":
        st.header("Head-to-Head Analysis")
        
        managers = sorted(teams_df['First Name'].unique())
        
        # Initialize random manager if not set
        if 'random_manager' not in st.session_state:
            st.session_state.random_manager = random.choice(managers)
        
        # Tabs for different H2H analyses
        tab1, tab2 = st.tabs(["💥 Direkter Vergleich", "🎯 Lieblings- & Angstgegner"])
        
        with tab1:
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
        
        with tab2:
            st.subheader("Lieblings- & Angstgegner Analyse")
            st.markdown("*Mindestanzahl: 5 Spiele gegeneinander*")
            
            # Manager selection with random button
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_manager = st.selectbox(
                    "Wähle Manager für Analyse:", 
                    managers, 
                    index=managers.index(st.session_state.random_manager) if st.session_state.random_manager in managers else 0,
                    key="opponent_analysis_manager"
                )
            with col2:
                if st.button("🎲 Zufällig", key="random_manager_btn"):
                    st.session_state.random_manager = random.choice([m for m in managers if m != selected_manager])
                    st.experimental_rerun()
            
            # Calculate and display opponent analysis
            favorites, nightmares = calculate_all_h2h_stats(processed_df, selected_manager, min_games=5)
            
            st.markdown(f"### Analyse für **{selected_manager}**")
            display_opponent_analysis(favorites, nightmares, selected_manager)
            
            # Optional: Show all H2H stats table
            if st.checkbox("Zeige alle Head-to-Head Statistiken", key="show_all_h2h"):
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
                    all_h2h_df = pd.DataFrame(all_opponents).sort_values('Siegquote', ascending=False)
                    st.dataframe(all_h2h_df, use_container_width=True, hide_index=True)
                else:
                    st.info("Keine Gegner mit mindestens 5 Spielen gefunden.")

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
                marker_color='#FFD700'
            ))
            
            fig.add_trace(go.Bar(
                name='🥈 Silver', 
                x=medal_table['Manager'],
                y=medal_table['Silver'],
                marker_color='#C0C0C0'
            ))
            
            fig.add_trace(go.Bar(
                name='🥉 Bronze',
                x=medal_table['Manager'],
                y=medal_table['Bronze'],
                marker_color='#CD7F32'
            ))
            
            fig.update_layout(
                title='Medal Distribution by Manager',
                xaxis_title='Manager',
                yaxis_title='Medal Count',
                barmode='stack',
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

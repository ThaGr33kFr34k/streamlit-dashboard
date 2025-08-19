import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

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

def calculate_playoff_stats(processed_df, teams_df):
    """Calculate playoff performance stats"""
    if processed_df is None or teams_df is None:
        return None
    
    managers = teams_df['First Name'].unique()
    playoff_stats = []
    
    for manager in managers:
        # Get all seasons this manager played
        manager_seasons = teams_df[teams_df['First Name'] == manager]['Year'].unique()
        seasons_played = len(manager_seasons)
        
        # Count playoff appearances (any non-regular season game)
        playoff_games = processed_df[
            (processed_df['Phase'] != 'NONE') & 
            (processed_df['Phase'] != 'Regular Season') &
            ((processed_df['Home_Manager'] == manager) | (processed_df['Away_Manager'] == manager))
        ]
        playoff_seasons = len(playoff_games['Season'].unique()) if len(playoff_games) > 0 else 0
        
        # Count championships (medals)
        championships = len(teams_df[(teams_df['First Name'] == manager) & (teams_df['Medal'] == 1)])
        finals = len(teams_df[(teams_df['First Name'] == manager) & (teams_df['Medal'].isin([1, 2]))])
        medals_total = len(teams_df[(teams_df['First Name'] == manager) & (teams_df['Medal'].isin([1, 2, 3]))])
        
        playoff_stats.append({
            'Manager': manager,
            'Seasons_Played': seasons_played,
            'Playoff_Appearances': playoff_seasons,
            'Championships': championships,
            'Finals': finals,
            'Total_Medals': medals_total,
            'Playoff_Rate': playoff_seasons / seasons_played if seasons_played > 0 else 0,
            'Championship_Rate': championships / seasons_played if seasons_played > 0 else 0
        })
    
    return pd.DataFrame(playoff_stats).sort_values('Championships', ascending=False)

def create_medal_overview(teams_df):
    """Create medal overview visualization"""
    if teams_df is None:
        return None
    
    medal_counts = teams_df.groupby(['First Name', 'Medal']).size().unstack(fill_value=0)
    
    # Ensure all medal types exist
    for medal in [1, 2, 3]:
        if medal not in medal_counts.columns:
            medal_counts[medal] = 0
    
    medal_counts = medal_counts[[1, 2, 3]]  # Reorder columns
    medal_counts.columns = ['🥇 Gold', '🥈 Silver', '🥉 Bronze']
    
    # Calculate total medals for sorting
    medal_counts['Total'] = medal_counts.sum(axis=1)
    medal_counts = medal_counts.sort_values('Total', ascending=False)
    
    return medal_counts.drop('Total', axis=1)

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
    
    # Sidebar
    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis",
        ["🥊 Head-to-Head", "🏆 Playoff Performance", "🏅 Medal Overview"]
    )
    
    # Main content based on selection
    if analysis_type == "🥊 Head-to-Head":
        st.header("Head-to-Head Analysis")
        
        managers = sorted(teams_df['First Name'].unique())
        
        col1, col2 = st.columns(2)
        with col1:
            manager1 = st.selectbox("Select Manager 1", managers, key="manager1")
        with col2:
            manager2 = st.selectbox("Select Manager 2", managers, key="manager2")
        
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
                    go.Bar(name=manager1, x=['Wins'], y=[h2h_stats['wins']], marker_color='#FF6B35'),
                    go.Bar(name=manager2, x=['Wins'], y=[h2h_stats['losses']], marker_color='#4ECDC4'),
                    go.Bar(name='Ties', x=['Wins'], y=[h2h_stats['ties']], marker_color='#45B7D1')
                ])
                
                fig.update_layout(
                    title=f"{manager1} vs {manager2} - Historical Matchup",
                    xaxis_title="Result",
                    yaxis_title="Games",
                    barmode='group'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No matchups found between {manager1} and {manager2}")
        else:
            st.info("Please select two different managers.")
    
    elif analysis_type == "🏆 Playoff Performance":
        st.header("Playoff Performance & Clutch Factor")
        
        playoff_stats = calculate_playoff_stats(processed_df, teams_df)
        
        if playoff_stats is not None:
            st.subheader("Clutch Factor Rankings")
            
            # Display table
            display_df = playoff_stats.copy()
            display_df['Playoff_Rate'] = display_df['Playoff_Rate'].apply(lambda x: f"{x:.1%}")
            display_df['Championship_Rate'] = display_df['Championship_Rate'].apply(lambda x: f"{x:.1%}")
            
            st.dataframe(
                display_df,
                column_config={
                    "Manager": "Manager",
                    "Seasons_Played": "Seasons",
                    "Playoff_Appearances": "Playoffs",
                    "Championships": "🏆 Titles",
                    "Finals": "Finals",
                    "Total_Medals": "🏅 Medals",
                    "Playoff_Rate": "Playoff %",
                    "Championship_Rate": "Title %"
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Visualization
            fig = px.scatter(
                playoff_stats,
                x='Playoff_Rate',
                y='Championship_Rate',
                size='Seasons_Played',
                hover_name='Manager',
                title='Clutch Factor: Playoff Success vs Championship Success',
                labels={
                    'Playoff_Rate': 'Playoff Appearance Rate',
                    'Championship_Rate': 'Championship Rate'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "🏅 Medal Overview":
        st.header("Medal Overview")
        
        medal_overview = create_medal_overview(teams_df)
        
        if medal_overview is not None:
            st.subheader("All-Time Medal Count")
            
            # Display medal table
            st.dataframe(
                medal_overview,
                use_container_width=True
            )
            
            # Medal visualization
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='🥇 Gold',
                x=medal_overview.index,
                y=medal_overview['🥇 Gold'],
                marker_color='#FFD700'
            ))
            
            fig.add_trace(go.Bar(
                name='🥈 Silver', 
                x=medal_overview.index,
                y=medal_overview['🥈 Silver'],
                marker_color='#C0C0C0'
            ))
            
            fig.add_trace(go.Bar(
                name='🥉 Bronze',
                x=medal_overview.index,
                y=medal_overview['🥉 Bronze'],
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
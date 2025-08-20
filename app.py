import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Data Loading
# -----------------------------
def load_data():
    try:
        teams_url = "YOUR_GOOGLE_SHEET_TEAMS_URL"
        matchups_url = "YOUR_GOOGLE_SHEET_MATCHUPS_URL"
        teams_df = pd.read_csv(teams_url)
        matchups_df = pd.read_csv(matchups_url)
        return teams_df, matchups_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# -----------------------------
# Helpers
# -----------------------------
def create_team_mapping(teams_df):
    return teams_df.set_index("TeamID")["First Name"].to_dict()

def process_matchup_data(matchups_df, team_mapping):
    try:
        df = matchups_df.copy()
        df["HOME"] = df["Home"].map(team_mapping)
        df["AWAY"] = df["Away"].map(team_mapping)
        return df
    except Exception as e:
        st.error(f"Error processing matchups: {e}")
        return None

# -----------------------------
# Playoff Stats
# -----------------------------
def calculate_playoff_stats(processed_df, teams_df):
    team_mapping = teams_df.set_index("TeamID")["First Name"].to_dict()
    managers = teams_df["First Name"].unique()
    playoff_stats = []

    for manager in managers:
        manager_seasons = teams_df[teams_df["First Name"] == manager]["Year"].unique()
        seasons_played = len(manager_seasons)
        manager_team_ids = teams_df[teams_df["First Name"] == manager]["TeamID"].unique()

        # Regular Season Games
        reg_games = processed_df[
            (processed_df["Phase"] == "Regular Season") &
            ((processed_df["HOME"].isin(manager_team_ids)) | (processed_df["AWAY"].isin(manager_team_ids)))
        ]
        reg_wins = ((reg_games["Winner"] == manager)).sum()
        reg_total = len(reg_games)
        reg_win_pct = reg_wins / reg_total if reg_total > 0 else 0

        # Playoff Games
        po_games = processed_df[
            (processed_df["Phase"] != "Regular Season") &
            (processed_df["Phase"] != "NONE") &
            ((processed_df["HOME"].isin(manager_team_ids)) | (processed_df["AWAY"].isin(manager_team_ids)))
        ]
        po_wins = ((po_games["Winner"] == manager)).sum()
        po_total = len(po_games)
        po_win_pct = po_wins / po_total if po_total > 0 else 0

        # Medals / Ranks
        championships = len(teams_df[(teams_df["First Name"] == manager) & (teams_df["Final Rank"] == 1)])
        finals = len(teams_df[(teams_df["First Name"] == manager) & (teams_df["Final Rank"].isin([1, 2]))])
        medals_total = len(teams_df[(teams_df["First Name"] == manager) & (teams_df["Final Rank"].isin([1, 2, 3]))])

        playoff_stats.append({
            "Manager": manager,
            "Seasons Played": seasons_played,
            "Regular Games": reg_total,
            "Regular Wins": reg_wins,
            "Regular Win%": reg_win_pct,
            "Playoff Games": po_total,
            "Playoff Wins": po_wins,
            "Playoff Win%": po_win_pct,
            "Championships": championships,
            "Finals": finals,
            "Total Medals": medals_total
        })

    df = pd.DataFrame(playoff_stats)

    reg_ranked = df.sort_values(by="Regular Win%", ascending=False).reset_index(drop=True)
    playoff_ranked = df.sort_values(by="Playoff Win%", ascending=False).reset_index(drop=True)

    return df, reg_ranked, playoff_ranked

# -----------------------------
# Medal Table
# -----------------------------
def create_medal_table(teams_df):
    managers = teams_df["First Name"].unique()
    medal_data = []
    for m in managers:
        gold = len(teams_df[(teams_df["First Name"] == m) & (teams_df["Final Rank"] == 1)])
        silver = len(teams_df[(teams_df["First Name"] == m) & (teams_df["Final Rank"] == 2)])
        bronze = len(teams_df[(teams_df["First Name"] == m) & (teams_df["Final Rank"] == 3)])
        medal_data.append({
            "Manager": m,
            "Gold": gold,
            "Silver": silver,
            "Bronze": bronze,
            "Total": gold + silver + bronze
        })
    df = pd.DataFrame(medal_data)
    df = df.sort_values(by=["Gold", "Silver", "Bronze"], ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df

# -----------------------------
# Main App
# -----------------------------
def main():
    st.markdown('<h1 class="main-header">🏀 Fantasy Basketball Analytics</h1>', unsafe_allow_html=True)

    with st.spinner("Loading data..."):
        teams_df, matchups_df = load_data()

    if teams_df is None or matchups_df is None:
        st.stop()

    team_mapping = create_team_mapping(teams_df)
    processed_df = process_matchup_data(matchups_df, team_mapping)
    if processed_df is None:
        st.stop()

    st.sidebar.title("Navigation")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis",
        ["🏆 Playoff Performance", "🥊 Head-to-Head", "🏅 Medal Overview"]
    )

    if analysis_type == "🏆 Playoff Performance":
        st.header("Playoff Performance & Clutch Factor")

        df, reg_ranked, playoff_ranked = calculate_playoff_stats(processed_df, teams_df)

        # Rankings side by side
        st.subheader("Ranking: Regular Season vs Playoffs")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Regular Season Ranking")
            st.dataframe(
                reg_ranked[["Manager", "Regular Games", "Regular Wins", "Regular Win%"]],
                hide_index=True,
                use_container_width=True
            )
        with col2:
            st.markdown("### Playoff Ranking")
            st.dataframe(
                playoff_ranked[["Manager", "Playoff Games", "Playoff Wins", "Playoff Win%"]],
                hide_index=True,
                use_container_width=True
            )

        # Gesamtübersicht
        st.subheader("Gesamtübersicht")
        styled = df.style.background_gradient(
            subset=["Regular Win%", "Playoff Win%"],
            cmap="RdYlGn",
            vmin=0,
            vmax=1
        ).format({
            "Regular Win%": "{:.3f}",
            "Playoff Win%": "{:.3f}"
        })
        st.dataframe(styled, use_container_width=True)

    elif analysis_type == "🥊 Head-to-Head":
        st.header("Head-to-Head Analysis")
        managers = sorted(teams_df["First Name"].unique())
        col1, col2 = st.columns(2)
        with col1:
            manager1 = st.selectbox("Select Manager 1", managers, key="m1")
        with col2:
            manager2 = st.selectbox("Select Manager 2", managers, key="m2")

        if manager1 == manager2:
            st.info("Please select two different managers.")
        else:
            # Simple h2h logic
            h2h_games = processed_df[
                ((processed_df["HOME"] == manager1) & (processed_df["AWAY"] == manager2)) |
                ((processed_df["HOME"] == manager2) & (processed_df["AWAY"] == manager1))
            ]
            wins1 = (h2h_games["Winner"] == manager1).sum()
            wins2 = (h2h_games["Winner"] == manager2).sum()
            ties = (h2h_games["Winner"] == "TIE").sum() if "TIE" in h2h_games["Winner"].values else 0
            total = len(h2h_games)

            if total > 0:
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1: st.metric("Total Games", total)
                with col2: st.metric(f"{manager1} Wins", wins1)
                with col3: st.metric(f"{manager2} Wins", wins2)
                with col4: st.metric("Ties", ties)
                with col5: st.metric(f"{manager1} Win %", f"{wins1/total:.1%}")
            else:
                st.info("No matchups found.")

    elif analysis_type == "🏅 Medal Overview":
        st.header("Medal Overview")
        medal_table = create_medal_table(teams_df)
        st.dataframe(medal_table, hide_index=True, use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(name="🥇 Gold", x=medal_table["Manager"], y=medal_table["Gold"], marker_color="#FFD700"))
        fig.add_trace(go.Bar(name="🥈 Silver", x=medal_table["Manager"], y=medal_table["Silver"], marker_color="#C0C0C0"))
        fig.add_trace(go.Bar(name="🥉 Bronze", x=medal_table["Manager"], y=medal_table["Bronze"], marker_color="#CD7F32"))
        fig.update_layout(title="Medal Distribution by Manager", barmode="stack", xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

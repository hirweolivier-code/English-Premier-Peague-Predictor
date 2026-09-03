import requests
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from supabase import create_client, Client
supabase: Client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)
def save_prediction_if_new(
    api_match_id,
    match_date,
    home_team,
    away_team,
    home_prob,
    draw_prob,
    away_prob,
    predicted_result
):
    existing = (
        supabase
        .table("predictions")
        .select("id")
        .eq("api_match_id", api_match_id)
        .execute()
    )

    if not existing.data:
        row = {
            "api_match_id": int(api_match_id),
            "match_date": str(match_date),
            "home_team": home_team,
            "away_team": away_team,
            "home_prob": float(home_prob),
            "draw_prob": float(draw_prob),
            "away_prob": float(away_prob),
            "predicted_result": predicted_result,
            "actual_result": None,
            "correct": None
        }

        supabase.table("predictions").insert(row).execute()
# ============================================================
# LOAD MODEL AND DATA
# ============================================================

project_path = "."

final_v2_model = joblib.load(
    f"{project_path}/final_v2_model.pkl"
)

features_v2 = joblib.load(
    f"{project_path}/features_v2.pkl"
)

football = pd.read_pickle(
    f"{project_path}/football_processed.pkl"
)
API_KEY = st.secrets["FOOTBALL_DATA_API_KEY"]

headers = {
    "X-Auth-Token": API_KEY
}
API_TO_MODEL_TEAM = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Leeds United FC": "Leeds",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",

    "Coventry City FC": "Coventry",
    "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich"
}

# ===================================================
# TEAM LOGOS
# ===================================================

team_logos = {
    "Arsenal": "https://resources.premierleague.com/premierleague/badges/50/t3.png",
    "Aston Villa": "https://resources.premierleague.com/premierleague/badges/50/t7.png",
    "Bournemouth": "https://resources.premierleague.com/premierleague/badges/50/t91.png",
    "Brighton": "https://resources.premierleague.com/premierleague/badges/50/t36.png",
    "Chelsea": "https://resources.premierleague.com/premierleague/badges/50/t8.png",
    "Liverpool": "https://resources.premierleague.com/premierleague/badges/50/t14.png",
    "Man City": "https://resources.premierleague.com/premierleague/badges/50/t43.png",
    "Man United": "https://resources.premierleague.com/premierleague/badges/50/t1.png",
    "Newcastle": "https://resources.premierleague.com/premierleague/badges/50/t4.png",
    "Tottenham": "https://resources.premierleague.com/premierleague/badges/50/t6.png"
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def resolve_team_name(team, df):

    teams = pd.unique(
        pd.concat([
            df["HomeTeam"],
            df["AwayTeam"]
        ])
    )

    matches = [
        t for t in teams
        if t.lower().strip() == team.lower().strip()
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise ValueError(
            f"Team '{team}' not found in dataset."
        )

    raise ValueError(
        f"Multiple matches found for '{team}': {matches}"
    )
def previous_h2h_stats(
    home_team,
    away_team,
    current_index,
    df,
    n=5
):

    previous_matches = df.iloc[:current_index]

    h2h = previous_matches[
        (
            (previous_matches["HomeTeam"] == home_team)
            &
            (previous_matches["AwayTeam"] == away_team)
        )
        |
        (
            (previous_matches["HomeTeam"] == away_team)
            &
            (previous_matches["AwayTeam"] == home_team)
        )
    ].tail(n)

    home_points = 0
    away_points = 0
    home_goals = 0
    away_goals = 0

    for _, match in h2h.iterrows():

        if match["HomeTeam"] == home_team:

            home_goals += match["FTHG"]
            away_goals += match["FTAG"]

            if match["FTR"] == "H":
                home_points += 3
            elif match["FTR"] == "A":
                away_points += 3
            else:
                home_points += 1
                away_points += 1

        else:

            home_goals += match["FTAG"]
            away_goals += match["FTHG"]

            if match["FTR"] == "A":
                home_points += 3
            elif match["FTR"] == "H":
                away_points += 3
            else:
                home_points += 1
                away_points += 1

    if len(h2h) == 0:
        return 0, 0, 0

    h2h_goal_diff = home_goals - away_goals

    return (
        home_points,
        away_points,
        h2h_goal_diff
    )

def previous_5_form(team, current_index, df):

    previous_matches = df.iloc[:current_index]

    team_matches = previous_matches[
        (previous_matches["HomeTeam"] == team) |
        (previous_matches["AwayTeam"] == team)
    ].tail(5)

    points = 0

    for _, match in team_matches.iterrows():

        if match["HomeTeam"] == team:

            if match["FTR"] == "H":
                points += 3

            elif match["FTR"] == "D":
                points += 1

        else:

            if match["FTR"] == "A":
                points += 3

            elif match["FTR"] == "D":
                points += 1

    return points


def previous_5_goal_stats(team, current_index, df):

    previous_matches = df.iloc[:current_index]

    team_matches = previous_matches[
        (previous_matches["HomeTeam"] == team) |
        (previous_matches["AwayTeam"] == team)
    ].tail(5)

    goals_for = []
    goals_against = []

    for _, match in team_matches.iterrows():

        if match["HomeTeam"] == team:

            goals_for.append(match["FTHG"])
            goals_against.append(match["FTAG"])

        else:

            goals_for.append(match["FTAG"])
            goals_against.append(match["FTHG"])

    if len(team_matches) == 0:
        return 0, 0

    return (
        np.mean(goals_for),
        np.mean(goals_against)
    )


def previous_5_location_form(team, current_index, df, location):

    previous_matches = df.iloc[:current_index]

    if location == "home":

        matches = previous_matches[
            previous_matches["HomeTeam"] == team
        ].tail(5)

        points = 0

        for _, match in matches.iterrows():

            if match["FTR"] == "H":
                points += 3

            elif match["FTR"] == "D":
                points += 1

    else:

        matches = previous_matches[
            previous_matches["AwayTeam"] == team
        ].tail(5)

        points = 0

        for _, match in matches.iterrows():

            if match["FTR"] == "A":
                points += 3

            elif match["FTR"] == "D":
                points += 1

    return points


def future_season_ppg(team, season, df):

    season_matches = df[
        df["Season"] == season
    ]

    team_matches = season_matches[
        (season_matches["HomeTeam"] == team)
        |
        (season_matches["AwayTeam"] == team)
    ]

    if len(team_matches) == 0:

        team_history = df[
            (df["HomeTeam"] == team)
            |
            (df["AwayTeam"] == team)
        ]

        if len(team_history) == 0:
            return 0.0

        latest_season = team_history.iloc[-1]["Season"]

        season_matches = df[
            df["Season"] == latest_season
        ]

        team_matches = season_matches[
            (season_matches["HomeTeam"] == team)
            |
            (season_matches["AwayTeam"] == team)
        ]

    if len(team_matches) == 0:
        return 0.0

    points = 0

    for _, match in team_matches.iterrows():

        if match["HomeTeam"] == team:

            if match["FTR"] == "H":
                points += 3

            elif match["FTR"] == "D":
                points += 1

        else:

            if match["FTR"] == "A":
                points += 3

            elif match["FTR"] == "D":
                points += 1

    return points / len(team_matches)


def previous_5_sot(team, current_index, df):

    previous_matches = df.iloc[:current_index]

    team_matches = previous_matches[
        (previous_matches["HomeTeam"] == team) |
        (previous_matches["AwayTeam"] == team)
    ]

    shots_on_target = []

    # Work backwards through matches
    for _, match in team_matches.iloc[::-1].iterrows():

        if match["HomeTeam"] == team:
            value = match.get("HST", np.nan)
        else:
            value = match.get("AST", np.nan)

        # Ignore API matches where SOT is unavailable
        if pd.notna(value):
            shots_on_target.append(value)

        if len(shots_on_target) == 5:
            break

    if len(shots_on_target) == 0:
        return 0

    return np.mean(shots_on_target)

def previous_5_sota(team, current_index, df):

    previous_matches = df.iloc[:current_index]

    team_matches = previous_matches[
        (previous_matches["HomeTeam"] == team) |
        (previous_matches["AwayTeam"] == team)
    ]

    shots_allowed = []

    # Work backwards through matches
    for _, match in team_matches.iloc[::-1].iterrows():

        if match["HomeTeam"] == team:
            value = match.get("AST", np.nan)
        else:
            value = match.get("HST", np.nan)

        if pd.notna(value):
            shots_allowed.append(value)

        if len(shots_allowed) == 5:
            break

    if len(shots_allowed) == 0:
        return 0

    return np.mean(shots_allowed)

def future_season_strength(team, season, df):

    previous = df[
        df["Season"] == season
    ]

    team_matches = previous[
        (previous["HomeTeam"] == team)
        |
        (previous["AwayTeam"] == team)
    ]

    if len(team_matches) == 0:

        team_history = df[
            (df["HomeTeam"] == team)
            |
            (df["AwayTeam"] == team)
        ]

        if len(team_history) == 0:
            return 1.0, 1.0

        latest_season = team_history.iloc[-1]["Season"]

        previous = df[
            df["Season"] == latest_season
        ]

        team_matches = previous[
            (previous["HomeTeam"] == team)
            |
            (previous["AwayTeam"] == team)
        ]

    if len(team_matches) == 0:
        return 1.0, 1.0

    goals_for = 0
    goals_against = 0

    for _, match in team_matches.iterrows():

        if match["HomeTeam"] == team:

            goals_for += match["FTHG"]
            goals_against += match["FTAG"]

        else:

            goals_for += match["FTAG"]
            goals_against += match["FTHG"]

    team_gf_per_game = (
        goals_for / len(team_matches)
    )

    team_ga_per_game = (
        goals_against / len(team_matches)
    )

    total_goals = (
        previous["FTHG"].sum()
        + previous["FTAG"].sum()
    )

    league_goals_per_team_match = (
        total_goals
        / (2 * len(previous))
    )

    attack_strength = (
        team_gf_per_game
        / league_goals_per_team_match
    )

    defense_strength = (
        team_ga_per_game
        / league_goals_per_team_match
    )

    return attack_strength, defense_strength


def get_current_elo_ratings(
    df,
    initial_rating=1500,
    k=40
):

    ratings = {}

    for _, row in df.iterrows():

        home = row["HomeTeam"]
        away = row["AwayTeam"]

        home_elo = ratings.get(
            home,
            initial_rating
        )

        away_elo = ratings.get(
            away,
            initial_rating
        )

        expected_home = 1 / (
            1 + 10 ** (
                (away_elo - home_elo) / 400
            )
        )

        expected_away = 1 - expected_home

        if row["FTR"] == "H":
            actual_home = 1.0
            actual_away = 0.0

        elif row["FTR"] == "A":
            actual_home = 0.0
            actual_away = 1.0

        else:
            actual_home = 0.5
            actual_away = 0.5

        ratings[home] = (
            home_elo
            + k * (actual_home - expected_home)
        )

        ratings[away] = (
            away_elo
            + k * (actual_away - expected_away)
        )

    return ratings
def get_upcoming_pl_fixtures():

    url = "https://api.football-data.org/v4/competitions/PL/matches"

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    fixtures = []

    for match in data["matches"]:

        if match["status"] in [
            "SCHEDULED",
            "TIMED"
        ]:

           fixtures.append({
               "id": match["id"],
               "Date": match["utcDate"],
               "Matchday": match["matchday"],
               "HomeTeam": match["homeTeam"]["name"],
               "AwayTeam": match["awayTeam"]["name"]
           })
         

    return pd.DataFrame(fixtures)

# ============================================================
# BUILD ALL 28 FEATURES
# ============================================================

def make_future_match_features_final(
    home_team,
    away_team,
    df,
    season=None
):

    home_team = resolve_team_name(home_team, df)
    away_team = resolve_team_name(away_team, df)

    if season is None:
        season = df.iloc[-1]["Season"]

    i = len(df)
    h2h_home_points, h2h_away_points, h2h_goal_diff = previous_h2h_stats(
    home_team,
    away_team,
    i,
    df,
    n=5)

    h_form = previous_5_form(home_team, i, df)
    a_form = previous_5_form(away_team, i, df)

    h_gf, h_ga = previous_5_goal_stats(home_team, i, df)
    a_gf, a_ga = previous_5_goal_stats(away_team, i, df)

    h_home_form = previous_5_location_form(
        home_team, i, df, "home"
    )

    a_away_form = previous_5_location_form(
        away_team, i, df, "away"
    )

    h_ppg = future_season_ppg(
        home_team, season, df
    )

    a_ppg = future_season_ppg(
        away_team, season, df
    )

    h_sot = previous_5_sot(home_team, i, df)
    a_sot = previous_5_sot(away_team, i, df)

    h_sota = previous_5_sota(home_team, i, df)
    a_sota = previous_5_sota(away_team, i, df)

    h_attack, h_defense = future_season_strength(
        home_team, season, df
    )

    a_attack, a_defense = future_season_strength(
        away_team, season, df
    )

    elo_ratings = get_current_elo_ratings(
        df,
        initial_rating=1500,
        k=40
    )

    h_elo = elo_ratings.get(home_team, 1500)
    a_elo = elo_ratings.get(away_team, 1500)

    future = pd.DataFrame([{
        "H2HHomePoints": h2h_home_points,
        "H2HAwayPoints": h2h_away_points,
        "H2HGoalDiff": h2h_goal_diff,

        "HomeForm5": h_form,
        "AwayForm5": a_form,

        "HomeGF5": h_gf,
        "HomeGA5": h_ga,
        "AwayGF5": a_gf,
        "AwayGA5": a_ga,

        "FormDiff": h_form - a_form,
        "AttackDiff": h_gf - a_gf,
        "DefenseDiff": a_ga - h_ga,

        "HomeHomeForm5": h_home_form,
        "AwayAwayForm5": a_away_form,

        "LocationFormDiff":
            h_home_form - a_away_form,

        "HomePPG": h_ppg,
        "AwayPPG": a_ppg,
        "PPGDiff": h_ppg - a_ppg,

        "HomeSOT5": h_sot,
        "AwaySOT5": a_sot,
        "SOTDiff": h_sot - a_sot,

        "HomeSOTA5": h_sota,
        "AwaySOTA5": a_sota,
        "SOTAllowedDiff": a_sota - h_sota,

        "HomeAttackStrength": h_attack,
        "HomeDefenseStrength": h_defense,

        "AwayAttackStrength": a_attack,
        "AwayDefenseStrength": a_defense,

        "AttackStrengthDiff":
            h_attack - a_attack,

        "DefenseStrengthDiff":
            a_defense - h_defense,

        "EloDiff40":
            h_elo - a_elo
    }])

    return future[features_v2]

def get_finished_pl_matches():

    url = (
        "https://api.football-data.org/v4/"
        "competitions/PL/matches?status=FINISHED"
    )

    response = requests.get(
        url,
        headers=headers,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    finished = []

    for match in data["matches"]:

        home_api = match["homeTeam"]["name"]
        away_api = match["awayTeam"]["name"]

        home = API_TO_MODEL_TEAM.get(
            home_api,
            home_api
        )

        away = API_TO_MODEL_TEAM.get(
            away_api,
            away_api
        )

        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        if (
            home_goals is None
            or away_goals is None
        ):
            continue

        if home_goals > away_goals:
            result = "H"

        elif home_goals < away_goals:
            result = "A"

        else:
            result = "D"

        finished.append({
            "api_match_id": match["id"],
            "Date": match["utcDate"],
            "Season": "2026/27",
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": result
        })

    return pd.DataFrame(finished)
st.markdown("---")
def update_finished_predictions(finished_matches):

    for _, match in finished_matches.iterrows():

        api_match_id = int(match["api_match_id"])
        actual_result = match["FTR"]

        existing = (
            supabase
            .table("predictions")
            .select("id,predicted_result,actual_result")
            .eq("api_match_id", api_match_id)
            .execute()
        )

        if not existing.data:
            continue

        row = existing.data[0]

         # Already updated before
        if row["actual_result"] is not None:
            continue

        predicted_result = row["predicted_result"]

        is_correct = predicted_result == actual_result

        (
            supabase
            .table("predictions")
            .update({
                "actual_result": actual_result,
                "correct": is_correct
            })
            .eq("api_match_id", api_match_id)
            .execute()
        )     
recent_results = get_finished_pl_matches()
update_finished_predictions(recent_results)

football_live = football.copy()

football_live["Date"] = pd.to_datetime(
    football_live["Date"],
    errors="coerce",
    utc=True
)

recent_results["Date"] = pd.to_datetime(
    recent_results["Date"],
    errors="coerce",
    utc=True
)

football_live = pd.concat(
    [
        football_live,
        recent_results
    ],
    ignore_index=True,
    sort=False
)

football_live = football_live.drop_duplicates(
    subset=[
        "Date",
        "HomeTeam",
        "AwayTeam"
    ],
    keep="last"
)

football_live = football_live.sort_values(
    "Date"
).reset_index(drop=True)
def get_live_model_performance():

    response = (
        supabase
        .table("predictions")
        .select("predicted_result,actual_result,correct")
        .not_.is_("actual_result", "null")
        .execute()
    )

    rows = response.data

    if not rows:
        return 0, 0, None

    completed = len(rows)
    correct = sum(1 for row in rows if row["correct"] is True)

    accuracy = correct / completed

    return completed, correct, accuracy
def get_prediction_history():

    response = (
        supabase
        .table("predictions")
        .select(
            "match_date,home_team,away_team,"
            "predicted_result,actual_result,correct"
        )
        .not_.is_("actual_result", "null")
        .order("match_date", desc=True)
        .execute()
    )

    return response.data

history = get_prediction_history()

st.subheader("📋 Prediction History")

if not history:
    st.info("No completed predictions yet.")

else:
    for row in history:

        if row["predicted_result"] == "H":
            predicted_text = row["home_team"]

        elif row["predicted_result"] == "A":
            predicted_text = row["away_team"]

        else:
            predicted_text = "Draw"

        if row["actual_result"] == "H":
            actual_text = row["home_team"]

        elif row["actual_result"] == "A":
            actual_text = row["away_team"]

        else:
            actual_text = "Draw"

        status = "✅" if row["correct"] else "❌"

        st.write(
            f"{status} {row['home_team']} vs {row['away_team']} "
            f"— Predicted: {predicted_text} | Actual: {actual_text}"
        )
# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="centered"
)

st.title("⚽ Premier League Match Predictor")
latest_update = football_live["Date"].max()

if pd.notna(latest_update):
    st.caption(
        "Data last updated: "
        + latest_update.strftime("%d %b %Y — %H:%M UTC")
    )
with st.expander("ℹ️ About the prediction model"):

    st.write(
        "Model: Multiclass Logistic Regression"
    )

    st.write(
        "Features: 31, including recent form, goals, "
        "home/away form, PPG, team strength, Elo and H2H."
    )

    st.write(
        "Model tuning: C = 0.003 using expanding "
        "time-series validation."
    )

    st.write(
        "Predictions are probabilities, not guaranteed results."
    )
st.write(
    "Select the home and away teams to predict the match outcome."
)
completed, correct, accuracy = get_live_model_performance()

st.subheader("📊 2026/27 Prediction Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Completed", completed)
col2.metric("Correct", correct)

if accuracy is None:
    col3.metric("Accuracy", "—")
else:
    col3.metric("Accuracy", f"{accuracy * 100:.1f}%")
teams = sorted(
    set(football["HomeTeam"])
    | set(football["AwayTeam"])
)

home_team = st.selectbox(
    "Home Team",
    teams
)

away_team = st.selectbox(
    "Away Team",
    teams,
    index=1
)

if st.button("Predict Match"):

    if home_team == away_team:

        st.error(
            "Home and away teams must be different."
        )

    else:

        X_future = make_future_match_features_final(
            home_team,
            away_team,
            football_live,
            season="2026/27"
        )

        probabilities = final_v2_model.predict_proba(
            X_future
        )[0]

        prob_dict = dict(
            zip(
                final_v2_model.classes_,
                probabilities
            )
        )

        p_home = prob_dict["H"]
        p_draw = prob_dict["D"]
        p_away = prob_dict["A"]

        prediction = max(
            prob_dict,
            key=prob_dict.get
        )

        col_home, col_vs, spacer, col_away = st.columns([1.8, 0.4,1.2, 1.8])

        with col_home:
            if home_team in team_logos:
                st.image(
                    team_logos[home_team],
                    width=75
                )

            st.markdown(
                f"<div style='text-align:left; font-size:24px; font-weight:700;'>"
                f"{home_team}</div>",
                unsafe_allow_html=True
            )

        with col_vs:
            st.markdown(
                "<div style='text-align:left; "
                "font-size:30px; font-weight:700; "
                "padding-top:35px;'>VS</div>",
                unsafe_allow_html=True
            )

        with col_away:
            
            if away_team in team_logos:
                st.image(
                    team_logos[away_team],
                    width=75
                )

            st.markdown(
                f"<div style='text-align:left; font-size:24px; font-weight:700;'>"
                f"{away_team}</div>",
                unsafe_allow_html=True
            )
        st.markdown("### Match probabilities")

        st.progress(
    float(p_home),
    text=f"{home_team} win — {p_home * 100:.1f}%"
)

        st.progress(
    float(p_draw),
    text=f"Draw — {p_draw * 100:.1f}%"
)

        st.progress(
    float(p_away),
    text=f"{away_team} win — {p_away * 100:.1f}%"
)    

        if prediction == "H":

            result_text = (
                f"🏆 {home_team} Win"
            )
        elif prediction == "A":

              result_text = (
                  f"🏆 {away_team} Win"
              )

        else:

            result_text = "🤝 Draw"

        st.success(
            f"Predicted Result: {result_text}"
        )

st.markdown("---")
st.header("📅 Upcoming Premier League Predictions")

try:

    upcoming_fixtures = get_upcoming_pl_fixtures()

    if len(upcoming_fixtures) == 0:

        st.info("No upcoming Premier League fixtures found.")

    else:

        
        upcoming_fixtures = upcoming_fixtures.sort_values("Date")

        next_matchday = upcoming_fixtures[
            "Matchday"
        ].dropna().min()

        upcoming_fixtures = upcoming_fixtures[
            upcoming_fixtures["Matchday"] == next_matchday
        ]

        st.caption(
            f"Premier League Matchweek {int(next_matchday)}"
        )

        for _, match in upcoming_fixtures.iterrows():

            home_api = match["HomeTeam"]
            away_api = match["AwayTeam"]

            home = API_TO_MODEL_TEAM.get(
                home_api,
                home_api
            )

            away = API_TO_MODEL_TEAM.get(
            away_api,
            away_api
            )

            model_teams = set(
                football["HomeTeam"]
            ) | set(
                football["AwayTeam"]
            )

            if (
                home not in model_teams
                or away not in model_teams
            ):
                continue

            X_future = make_future_match_features_final(
                home,
                away,
                football_live,
                season="2026/27"
            )

            probabilities = final_v2_model.predict_proba(
                X_future
            )[0]

            prob_dict = dict(
                zip(
                    final_v2_model.classes_,
                    probabilities
                )
            )

            p_home = prob_dict["H"]
            p_draw = prob_dict["D"]
            p_away = prob_dict["A"]

            prediction = max(
                prob_dict,
                key=prob_dict.get
            )

            if prediction == "H":
                prediction_text = home

            elif prediction == "A":
                prediction_text = away

            else:
                prediction_text = "Draw"

            match_date = pd.to_datetime(
                match["Date"]
            )
            save_prediction_if_new(
                api_match_id=match["id"],
                match_date=match_date.isoformat(),
                home_team=home,
                away_team=away,
                home_prob=p_home,
                draw_prob=p_draw,
                away_prob=p_away,
                predicted_result=prediction
            )
            st.caption(
                match_date.strftime(
                    "%a %d %b %Y — %H:%M UTC"
                )
            )
            
            st.markdown(
                f"### {home} vs {away}"
        )

            st.write(
                f"Home: {p_home * 100:.1f}% | "
                f"Draw: {p_draw * 100:.1f}% | "
                f"Away: {p_away * 100:.1f}%"
            )

            st.success(
                f"Prediction: {prediction_text}"
            )
except Exception as e:
    st.error(
        f"Could not load upcoming fixtures: {type(e).__name__}: {e}"
    )

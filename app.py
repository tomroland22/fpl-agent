"""
FPL Agent - single-file dashboard (deploy via Streamlit Community Cloud).
Everything in one file on purpose, so it's easy to upload from a phone.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="FPL Agent", page_icon="⚽", layout="wide")
POSITION_COLORS = {"GKP": "#7c3aed", "DEF": "#2563eb", "MID": "#16a34a", "FWD": "#dc2626"}
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
BASE_URL = "https://fantasy.premierleague.com/api"

# ---------------------------------------------------------------------------
# FPL API client
# ---------------------------------------------------------------------------
class FPLClient:
    def __init__(self, timeout=15):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "fpl-agent/0.1"})
        self.timeout = timeout

    def _get(self, path, params=None):
        r = self.session.get(f"{BASE_URL}/{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def bootstrap_static(self):
        return self._get("bootstrap-static/")

    def entry(self, entry_id):
        return self._get(f"entry/{entry_id}/")

    def entry_picks(self, entry_id, event):
        return self._get(f"entry/{entry_id}/event/{event}/picks/")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def players_dataframe(bootstrap):
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    df = pd.DataFrame(bootstrap["elements"])
    df["team_name"] = df["team"].map(teams)
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["price"] = df["now_cost"] / 10.0
    df["form"] = pd.to_numeric(df["form"], errors="coerce")
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce")
    df["points_per_million"] = (df["total_points"] / df["price"]).round(2)
    cols = ["id", "web_name", "team_name", "position", "price", "total_points",
            "points_per_game", "form", "selected_by_percent", "points_per_million",
            "minutes", "status"]
    return df[cols].sort_values("total_points", ascending=False).reset_index(drop=True)


def squad_from_picks(picks, players_df):
    pick_rows = pd.DataFrame(picks["picks"])
    merged = pick_rows.merge(players_df, left_on="element", right_on="id", how="left")
    merged["role"] = merged.apply(
        lambda r: "Captain" if r["is_captain"] else ("Vice" if r["is_vice_captain"] else ""),
        axis=1,
    )
    return merged.sort_values(["multiplier", "total_points"], ascending=[False, False])


def top_value_picks(players_df, min_minutes=300, n=15):
    df = players_df[players_df["minutes"] >= min_minutes]
    return df.sort_values("points_per_million", ascending=False).head(n)


def suggest_transfers(squad_df, players_df, bank, n_suggestions=3):
    suggestions = []
    for _, player in squad_df.iterrows():
        budget = player["price"] + bank
        candidates = players_df[
            (players_df["position"] == player["position"])
            & (players_df["price"] <= budget)
            & (players_df["id"] != player["id"])
            & (players_df["status"] == "a")
        ]
        better = candidates[candidates["total_points"] > player["total_points"]]
        if not better.empty:
            best = better.sort_values("total_points", ascending=False).iloc[0]
            suggestions.append({
                "out": player["web_name"], "out_points": player["total_points"],
                "in": best["web_name"], "in_points": best["total_points"],
                "in_price": best["price"],
                "cost_change": round(best["price"] - player["price"], 1),
            })
    if not suggestions:
        return pd.DataFrame()
    return pd.DataFrame(suggestions).sort_values("in_points", ascending=False).head(n_suggestions)


# ---------------------------------------------------------------------------
# Synthetic demo data (works with no network access)
# ---------------------------------------------------------------------------
def generate_demo_players_df(n_players=220, seed=42):
    rng = np.random.default_rng(seed)
    teams = ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
             "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds",
             "Liverpool", "Man City", "Man Utd", "Newcastle", "Nott'm Forest",
             "Sunderland", "Spurs", "West Ham", "Wolves", "Burnley"]
    first = ["J.", "M.", "L.", "K.", "D.", "R.", "A.", "S.", "T.", "B."]
    surnames = ["Silva", "Costa", "Johnson", "Mbeki", "Rahman", "Novak", "Fischer",
                "Diallo", "Petrov", "Okafor", "Larsen", "Ricci", "Haaland", "Saka",
                "Foden", "Palmer", "Bruno", "Salah", "Watkins", "Isak"]

    positions = rng.choice(list(POSITION_MAP.values()), size=n_players, p=[0.12, 0.35, 0.35, 0.18])
    team_col = rng.choice(teams, size=n_players)
    quality = rng.beta(2, 5, size=n_players)
    price_base = {"GKP": 4.5, "DEF": 4.5, "MID": 5.5, "FWD": 5.5}
    price_range = {"GKP": 3.0, "DEF": 4.5, "MID": 8.5, "FWD": 8.0}
    points_scale = {"GKP": 140, "DEF": 150, "MID": 220, "FWD": 210}

    price = np.array([price_base[p] + quality[i] * price_range[p]
                       for i, p in enumerate(positions)]) + rng.normal(0, 0.3, n_players)
    price = np.clip(price, 3.9, 15.5).round(1)
    points = np.array([quality[i] * points_scale[p]
                        for i, p in enumerate(positions)]) + rng.normal(0, 12, n_players)
    points = np.clip(points, 0, None).round().astype(int)
    minutes = np.clip(quality * 3200 + rng.normal(0, 400, n_players), 0, 3400).astype(int)
    form = np.clip(quality * 8 + rng.normal(0, 1, n_players), 0, 9).round(1)
    ownership = np.clip((quality ** 2) * 55 + rng.normal(0, 4, n_players), 0.1, 70).round(1)
    names = [f"{rng.choice(first)}{rng.choice(surnames)}" for _ in range(n_players)]

    df = pd.DataFrame({
        "id": range(1, n_players + 1), "web_name": names, "team_name": team_col,
        "position": positions, "price": price, "total_points": points,
        "points_per_game": (points / np.clip(minutes / 90, 1, None)).round(2),
        "form": form, "selected_by_percent": ownership, "minutes": minutes,
        "status": "a",
    })
    df["points_per_million"] = (df["total_points"] / df["price"]).round(2)
    return df.sort_values("total_points", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_demo_data():
    return generate_demo_players_df()


@st.cache_data(ttl=3600)
def load_live_data():
    bootstrap = FPLClient().bootstrap_static()
    return players_dataframe(bootstrap)


@st.cache_data(ttl=600)
def load_squad(entry_id, gw, _players_df):
    client = FPLClient()
    picks = client.entry_picks(entry_id, gw)
    entry_info = client.entry(entry_id)
    bank = entry_info.get("last_deadline_bank", 0) / 10.0
    return squad_from_picks(picks, _players_df), bank


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.sidebar.title("⚽ FPL Agent")
data_mode = st.sidebar.radio("Data source", ["Demo (offline)", "Live"], index=0)
entry_id = st.sidebar.text_input("Your FPL entry ID", value="501017")
gw = st.sidebar.number_input("Gameweek", min_value=1, max_value=38, value=1)
min_minutes = st.sidebar.slider("Minimum minutes played", 0, 3000, 300, step=100)

squad_df, bank, squad_names = None, None, None

if data_mode == "Demo (offline)":
    players_df = load_demo_data()
    st.sidebar.info("Synthetic data. Switch to Live for your real squad and prices.")
else:
    try:
        players_df = load_live_data()
        if entry_id:
            try:
                squad_df, bank = load_squad(int(entry_id), int(gw), players_df)
                squad_names = squad_df["web_name"].tolist()
            except Exception as e:
                st.sidebar.warning(f"Couldn't load squad picks yet: {e}")
    except Exception as e:
        st.error(f"Couldn't reach the live FPL API: {e}")
        st.stop()

filtered = players_df[players_df["minutes"] >= min_minutes]

st.title("FPL Agent Dashboard")
st.caption(f"{len(filtered)} players shown (min {min_minutes} minutes played)")

if squad_df is not None:
    st.subheader("Your squad")
    st.dataframe(
        squad_df[["web_name", "position", "team_name", "price", "total_points", "role"]],
        width="stretch", hide_index=True,
    )
    suggestions = suggest_transfers(squad_df, players_df, bank)
    if not suggestions.empty:
        st.subheader(f"Possible upgrades (bank: £{bank}m)")
        st.dataframe(suggestions, width="stretch", hide_index=True)

st.subheader("Price vs Points")
fig = px.scatter(
    filtered, x="price", y="total_points", color="position",
    color_discrete_map=POSITION_COLORS,
    hover_data=["web_name", "team_name", "form", "selected_by_percent"],
    labels={"price": "Price (£m)", "total_points": "Total points"},
)
if squad_names:
    owned = filtered[filtered["web_name"].isin(squad_names)]
    fig.add_scatter(
        x=owned["price"], y=owned["total_points"], mode="markers",
        marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
        name="Your squad", hoverinfo="skip",
    )
fig.update_layout(height=550)
st.plotly_chart(fig, width="stretch")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Form vs Ownership")
    fig2 = px.scatter(
        filtered, x="selected_by_percent", y="form", color="position",
        color_discrete_map=POSITION_COLORS,
        hover_data=["web_name", "team_name", "total_points"],
        labels={"selected_by_percent": "Selected by (%)", "form": "Form"},
    )
    st.plotly_chart(fig2, width="stretch")

with col2:
    st.subheader("Top value picks (pts / £m)")
    top_value = top_value_picks(players_df, min_minutes=min_minutes, n=15)
    fig3 = px.bar(
        top_value.sort_values("points_per_million"),
        x="points_per_million", y="web_name", color="position",
        color_discrete_map=POSITION_COLORS, orientation="h",
        labels={"points_per_million": "Pts / £m", "web_name": ""},
    )
    st.plotly_chart(fig3, width="stretch")

st.divider()
st.caption("Heuristic suggestions only - doesn't yet account for fixture runs, "
           "price-change timing, or remaining chips.")

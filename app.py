"""
FPL Agent - single-file dashboard (deploy via Streamlit Community Cloud).
Everything in one file on purpose, so it's easy to upload/edit from a phone.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="FPL Agent", page_icon="⚽", layout="wide")

POSITION_COLORS = {"GKP": "#a78bfa", "DEF": "#60a5fa", "MID": "#34d399", "FWD": "#fb7185"}
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]
STATUS_LABELS = {"a": "Available", "d": "Doubtful", "i": "Injured", "s": "Suspended",
                  "u": "Unavailable", "n": "Not available"}
BASE_URL = "https://fantasy.premierleague.com/api"
LAST_SEASON_CSV = ("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
                    "master/data/2025-26/players_raw.csv")
FONT = "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
K_GAMEWEEKS = 8       # gameweeks until "this season" fully outweighs "last season" in Blended mode
LOW_SAMPLE_STARTS = 3  # fewer starts than this this season = flagged as a small sample


# ---------------------------------------------------------------------------
# Chart styling helpers - applied to every chart so they look consistent
# ---------------------------------------------------------------------------
def style_fig(fig, height=440, show_legend=True):
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT, size=12, color="#374151"),
        title_font=dict(family=FONT, size=15, color="#111827"),
        margin=dict(l=8, r=8, t=40, b=8),
        height=height,
        showlegend=show_legend,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#e5e7eb"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f3f4f6", zeroline=False, linecolor="#e5e7eb")
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor="#e5e7eb")
    return fig


def style_bar(fig, n_items=10):
    fig.update_traces(marker_line_width=0)
    return style_fig(fig, height=90 + 34 * max(n_items, 1), show_legend=False)


def style_scatter(fig, height=520):
    fig.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=0.5, color="white")))
    return style_fig(fig, height=height)


def show(fig):
    """Every chart goes through here - keeps the mobile modebar off everywhere."""
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_ranked_bar(df, score_col, x_label, sort_mode, n=12):
    """Shared renderer for every 'top N players by some score' bar chart,
    with the Score/Position sort toggle applied consistently."""
    d = df.nlargest(n, score_col).copy()
    if d.empty:
        st.caption("Not enough data for this view yet.")
        return
    if sort_mode == "Position":
        d["position"] = pd.Categorical(d["position"], categories=POSITION_ORDER, ordered=True)
        d = d.sort_values(["position", score_col])
    else:
        d = d.sort_values(score_col)
    fig = px.bar(d, x=score_col, y="web_name", color="position", color_discrete_map=POSITION_COLORS,
                 orientation="h", labels={score_col: x_label, "web_name": ""})
    fig.update_yaxes(categoryorder="array", categoryarray=d["web_name"].tolist())
    show(style_bar(fig, len(d)))


# ---------------------------------------------------------------------------
# FPL API client
# ---------------------------------------------------------------------------
class FPLClient:
    def __init__(self, timeout=15):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "fpl-agent/0.3"})
        self.timeout = timeout

    def _get(self, path, params=None):
        r = self.session.get(f"{BASE_URL}/{path}", params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def bootstrap_static(self):
        return self._get("bootstrap-static/")

    def fixtures(self):
        return self._get("fixtures/")

    def player_summary(self, player_id):
        return self._get(f"element-summary/{player_id}/")

    def entry(self, entry_id):
        return self._get(f"entry/{entry_id}/")

    def entry_picks(self, entry_id, event):
        return self._get(f"entry/{entry_id}/event/{event}/picks/")


# ---------------------------------------------------------------------------
# Last season lookup - the "value" baseline before this season has points
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_last_season_lookup():
    """Best-effort fetch of last season's final points, keyed by player name.
    Returns {} on any failure so the rest of the app degrades gracefully."""
    try:
        df = pd.read_csv(LAST_SEASON_CSV)
        key = (df["first_name"].fillna("") + " " + df["second_name"].fillna("")).str.lower().str.strip()
        return df.assign(key=key).groupby("key")["total_points"].max().to_dict()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Gameweek helpers
# ---------------------------------------------------------------------------
def gameweek_progress(events):
    """Returns (completed_gameweeks, current_event_id)."""
    completed = sum(1 for e in events if e.get("finished"))
    upcoming = next((e["id"] for e in events if not e.get("finished")), None)
    current_event = upcoming if upcoming is not None else (events[-1]["id"] if events else 1)
    return completed, current_event


# ---------------------------------------------------------------------------
# Fixture difficulty
# ---------------------------------------------------------------------------
def fixture_difficulty_table(fixtures, teams_map, current_event, n=5):
    rows = []
    for f in fixtures:
        ev = f.get("event")
        if ev is None or f.get("finished") or not (current_event <= ev < current_event + n):
            continue
        rows.append({"team_id": f["team_h"], "difficulty": f["team_h_difficulty"], "event": ev})
        rows.append({"team_id": f["team_a"], "difficulty": f["team_a_difficulty"], "event": ev})
    if not rows:
        return pd.DataFrame(columns=["team_name", "avg_difficulty", "fixtures_count"])
    df = pd.DataFrame(rows)
    agg = df.groupby("team_id").agg(avg_difficulty=("difficulty", "mean"),
                                     fixtures_count=("difficulty", "count")).reset_index()
    agg["team_name"] = agg["team_id"].map(teams_map)
    return agg.sort_values("avg_difficulty").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core player data
# ---------------------------------------------------------------------------
def players_dataframe(bootstrap):
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    df = pd.DataFrame(bootstrap["elements"])
    df["team_name"] = df["team"].map(teams)
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["price"] = df["now_cost"] / 10.0
    df["form"] = pd.to_numeric(df["form"], errors="coerce")
    df["points_per_game"] = pd.to_numeric(df["points_per_game"], errors="coerce")
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce")
    df["ep_next"] = pd.to_numeric(df.get("ep_next"), errors="coerce")
    df["cost_change_start"] = df.get("cost_change_start", 0) / 10.0
    df["cost_change_event"] = df.get("cost_change_event", 0) / 10.0
    df["transfers_in"] = pd.to_numeric(df.get("transfers_in", 0), errors="coerce")
    df["transfers_out"] = pd.to_numeric(df.get("transfers_out", 0), errors="coerce")
    df["transfers_balance"] = df["transfers_in"] - df["transfers_out"]
    df["starts"] = pd.to_numeric(df.get("starts", 0), errors="coerce").fillna(0)
    df["news"] = df.get("news", "").fillna("")

    completed_gameweeks, current_event = gameweek_progress(bootstrap["events"])
    df["completed_gameweeks"] = completed_gameweeks
    df["current_event"] = current_event

    lookup = load_last_season_lookup()
    key = (df["first_name"].fillna("") + " " + df["second_name"].fillna("")).str.lower().str.strip()
    df["last_season_points"] = key.map(lookup)

    df["is_new"] = df["last_season_points"].isna() & (df["minutes"] == 0)
    df["reliability"] = np.where(df["starts"] < LOW_SAMPLE_STARTS, "Low sample", "OK")
    df["trend"] = np.select(
        [df["form"] > df["points_per_game"] + 0.5, df["form"] < df["points_per_game"] - 0.5],
        ["Rising", "Falling"], default="Stable",
    )

    cols = ["id", "web_name", "team_name", "position", "price", "total_points",
            "last_season_points", "is_new", "completed_gameweeks", "current_event",
            "points_per_game", "form", "trend", "selected_by_percent", "reliability",
            "starts", "minutes", "status", "news", "ep_next", "cost_change_start",
            "cost_change_event", "transfers_in", "transfers_out", "transfers_balance"]
    return df[cols].reset_index(drop=True)


def apply_value_basis(df, mode="blended"):
    """Computes value_basis (a season-scale points figure) three ways:
    - 'last_season': just last season's total
    - 'this_season': just this season's total (0 pre-season, by design)
    - 'blended' (default): shrinks toward last season's per-gameweek rate
      early on, and toward this season's actual rate as more gameweeks
      complete - avoids one lucky/unlucky gameweek swinging everything.
    """
    df = df.copy()
    cgw = int(df["completed_gameweeks"].iloc[0]) if len(df) else 0
    last_rate = df["last_season_points"] / 38.0
    this_rate = df["total_points"] / cgw if cgw > 0 else pd.Series(np.nan, index=df.index)

    if mode == "last_season":
        basis = df["last_season_points"]
    elif mode == "this_season":
        basis = df["total_points"] if cgw > 0 else pd.Series(np.nan, index=df.index)
    else:
        weight = min(cgw / K_GAMEWEEKS, 1.0)
        both = this_rate.notna() & last_rate.notna()
        rate = pd.Series(np.nan, index=df.index)
        rate[both] = weight * this_rate[both] + (1 - weight) * last_rate[both]
        only_this = this_rate.notna() & ~both
        rate[only_this] = this_rate[only_this]
        only_last = last_rate.notna() & ~both
        rate[only_last] = last_rate[only_last]
        basis = rate * 38

    basis = basis.where(~df["is_new"])  # never fabricate a value for a genuine unknown
    df["value_basis"] = basis
    df["value_per_million"] = (df["value_basis"] / df["price"]).round(2)
    return df


def squad_from_picks(picks, players_df):
    pick_rows = pd.DataFrame(picks["picks"])
    merged = pick_rows.merge(players_df, left_on="element", right_on="id", how="left")
    merged["role"] = merged.apply(
        lambda r: "Captain" if r["is_captain"] else ("Vice" if r["is_vice_captain"] else ""),
        axis=1,
    )
    return merged.sort_values(["multiplier", "value_basis"], ascending=[False, False])


def top_value_picks(players_df, min_minutes=0, n=15):
    df = players_df[(players_df["minutes"] >= min_minutes) & (~players_df["is_new"])]
    return df.sort_values("value_per_million", ascending=False).head(n)


def add_percentiles(df):
    """Within-position percentile ranks for the Smart Picks views."""
    df = df.copy()
    valid = ~df["is_new"]
    for col in ["price_pct", "value_pct", "own_pct"]:
        df[col] = np.nan
    if valid.any():
        df.loc[valid, "price_pct"] = df.loc[valid].groupby("position")["price"].rank(pct=True) * 100
        df.loc[valid, "value_pct"] = df.loc[valid].groupby("position")["value_basis"].rank(pct=True) * 100
        df.loc[valid, "own_pct"] = df.loc[valid].groupby("position")["selected_by_percent"].rank(pct=True) * 100
    df["overpriced_score"] = df["price_pct"] - df["value_pct"]
    df["underpriced_score"] = df["value_pct"] - df["price_pct"]
    df["overowned_score"] = df["own_pct"] - df["value_pct"]
    df["underowned_score"] = df["value_pct"] - df["own_pct"]
    return df


def suggest_transfers(squad_df, players_df, bank, fixture_lookup=None, n_suggestions=3):
    fixture_lookup = fixture_lookup or {}

    def adjusted(row):
        base = row["value_basis"]
        if pd.isna(base):
            return np.nan
        diff = fixture_lookup.get(row["team_name"], 3.0)  # 3 = neutral difficulty
        return base - (diff - 3.0) * 3.0  # easier run nudges the score up, harder nudges down

    suggestions = []
    for _, player in squad_df.iterrows():
        if pd.isna(player.get("value_basis")):
            continue
        budget = player["price"] + bank
        candidates = players_df[
            (players_df["position"] == player["position"])
            & (players_df["price"] <= budget)
            & (players_df["id"] != player["id"])
            & (players_df["status"] == "a")
            & (~players_df["is_new"])
        ].copy()
        if candidates.empty:
            continue
        candidates["adj_score"] = candidates.apply(adjusted, axis=1)
        player_adj = adjusted(player)
        better = candidates[candidates["adj_score"] > player_adj]
        if not better.empty:
            best = better.sort_values("adj_score", ascending=False).iloc[0]
            suggestions.append({
                "out": player["web_name"], "out_pts": round(player["value_basis"], 0),
                "in": best["web_name"], "in_pts": round(best["value_basis"], 0),
                "in_price": best["price"],
                "cost_change": round(best["price"] - player["price"], 1),
            })
    if not suggestions:
        return pd.DataFrame()
    return pd.DataFrame(suggestions).sort_values("in_pts", ascending=False).head(n_suggestions)


def player_history_df(client, player_id):
    data = client.player_summary(player_id)
    hist = pd.DataFrame(data.get("history", []))
    if hist.empty:
        return hist
    hist = hist.rename(columns={"round": "gameweek", "total_points": "points", "value": "price_x10"})
    hist["price"] = hist["price_x10"] / 10.0
    return hist[["gameweek", "points", "minutes", "price"]]


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
    points_per_game = np.clip(form + rng.normal(0, 1.2, n_players), 0, 9).round(1)
    ownership = np.clip((quality ** 2) * 55 + rng.normal(0, 4, n_players), 0.1, 70).round(1)
    ep_next = np.clip(quality * 8 + rng.normal(0, 1, n_players), 0, 12).round(1)
    cost_change_start = np.round(
        np.where(rng.random(n_players) < 0.25,
                 (quality - 0.3) * rng.uniform(0.5, 2.0, n_players), 0.0), 1)
    cost_change_event = np.round(cost_change_start * rng.uniform(0.1, 0.4, n_players), 1)
    transfers_in = np.clip((quality ** 2) * 800000 + rng.normal(0, 30000, n_players), 0, None).astype(int)
    transfers_out = np.clip((1 - quality) * 300000 + rng.normal(0, 20000, n_players), 0, None).astype(int)
    starts = np.clip((minutes / 75).round(), 0, 10).astype(int)
    is_new = rng.random(n_players) < 0.04
    status = np.where(rng.random(n_players) < 0.06, rng.choice(["i", "d", "s"], n_players), "a")
    news = np.where(status != "a", "Knock picked up in training - assessed ahead of next match.", "")
    names = [f"{rng.choice(first)}{rng.choice(surnames)}" for _ in range(n_players)]

    df = pd.DataFrame({
        "id": range(1, n_players + 1), "web_name": names, "team_name": team_col,
        "position": positions, "price": price, "total_points": points,
        "last_season_points": np.where(is_new, np.nan,
                                        np.clip(points + rng.normal(0, 15, n_players), 0, None).round()),
        "points_per_game": points_per_game, "form": form, "selected_by_percent": ownership,
        "minutes": minutes, "starts": starts, "status": status, "news": news,
        "ep_next": ep_next, "cost_change_start": cost_change_start, "cost_change_event": cost_change_event,
        "transfers_in": transfers_in, "transfers_out": transfers_out, "is_new": is_new,
    })
    df["transfers_balance"] = df["transfers_in"] - df["transfers_out"]
    df["completed_gameweeks"] = 10  # demo simulates mid-season
    df["current_event"] = 11
    df["reliability"] = np.where(df["starts"] < LOW_SAMPLE_STARTS, "Low sample", "OK")
    df["trend"] = np.select(
        [df["form"] > df["points_per_game"] + 0.5, df["form"] < df["points_per_game"] - 0.5],
        ["Rising", "Falling"], default="Stable",
    )
    return df.reset_index(drop=True)


def generate_demo_fixture_difficulty():
    teams = ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
             "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds",
             "Liverpool", "Man City", "Man Utd", "Newcastle", "Nott'm Forest",
             "Sunderland", "Spurs", "West Ham", "Wolves", "Burnley"]
    rng = np.random.default_rng(7)
    diff = rng.uniform(1.8, 4.2, len(teams)).round(2)
    return pd.DataFrame({"team_name": teams, "avg_difficulty": diff,
                          "fixtures_count": 5}).sort_values("avg_difficulty").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_demo_data():
    return generate_demo_players_df()


@st.cache_data(ttl=3600)
def load_live_bootstrap():
    return FPLClient().bootstrap_static()


@st.cache_data(ttl=3600)
def load_live_fixtures():
    return FPLClient().fixtures()


@st.cache_data(ttl=600)
def load_squad(entry_id, gw, _players_df):
    client = FPLClient()
    picks = client.entry_picks(entry_id, gw)
    entry_info = client.entry(entry_id)
    bank = entry_info.get("last_deadline_bank", 0) / 10.0
    return squad_from_picks(picks, _players_df), bank


@st.cache_data(ttl=600)
def load_player_history(player_id):
    return player_history_df(FPLClient(), player_id)


# ---------------------------------------------------------------------------
# Sidebar: data source, value basis, filters
# ---------------------------------------------------------------------------
st.sidebar.title("⚽ FPL Agent")
data_mode = st.sidebar.radio("Data source", ["Demo (offline)", "Live"], index=0)
entry_id = st.sidebar.text_input("Your FPL entry ID", value="501017")
gw = st.sidebar.number_input("Gameweek", min_value=1, max_value=38, value=1)
value_mode_label = st.sidebar.radio("Value basis", ["Blended", "Last season only", "This season only"], index=0)
value_mode = {"Blended": "blended", "Last season only": "last_season",
              "This season only": "this_season"}[value_mode_label]

squad_df, bank, squad_names, fixture_diff = None, None, None, pd.DataFrame()
teams_map, current_event = {}, 1

if data_mode == "Demo (offline)":
    raw_players_df = load_demo_data()
    fixture_diff = generate_demo_fixture_difficulty()
    st.sidebar.info("Synthetic data. Switch to Live for your real squad and prices.")
else:
    try:
        bootstrap = load_live_bootstrap()
        raw_players_df = players_dataframe(bootstrap)
        teams_map = {t["id"]: t["name"] for t in bootstrap["teams"]}
        current_event = raw_players_df["current_event"].iloc[0] if len(raw_players_df) else 1
        try:
            fixtures_raw = load_live_fixtures()
            fixture_diff = fixture_difficulty_table(fixtures_raw, teams_map, current_event)
        except Exception as e:
            st.sidebar.warning(f"Couldn't load fixtures: {e}")
        if entry_id:
            try:
                players_df_tmp = apply_value_basis(raw_players_df, value_mode)
                squad_df, bank = load_squad(int(entry_id), int(gw), players_df_tmp)
                squad_names = squad_df["web_name"].tolist()
            except Exception as e:
                st.sidebar.warning(f"Couldn't load squad picks yet: {e}")
    except Exception as e:
        st.error(f"Couldn't reach the live FPL API: {e}")
        st.stop()

players_df = apply_value_basis(raw_players_df, value_mode)
if squad_df is not None:
    squad_df = apply_value_basis(
        squad_df.drop(columns=["value_basis", "value_per_million"], errors="ignore"), value_mode
    )
fixture_lookup = dict(zip(fixture_diff["team_name"], fixture_diff["avg_difficulty"])) if len(fixture_diff) else {}
players_df["fixture_difficulty"] = players_df["team_name"].map(fixture_lookup).fillna(3.0)

completed_gameweeks = int(players_df["completed_gameweeks"].iloc[0]) if len(players_df) else 0

st.sidebar.divider()
position_filter = st.sidebar.multiselect("Position", ["GKP", "DEF", "MID", "FWD"])
team_filter = st.sidebar.multiselect("Team", sorted(players_df["team_name"].dropna().unique()))
own_range = st.sidebar.slider("Ownership % range", 0.0, 100.0, (0.0, 100.0))
min_minutes = st.sidebar.slider("Minimum minutes played", 0, 3400, 0, step=100)

filtered = players_df[
    (players_df["minutes"] >= min_minutes)
    & (players_df["selected_by_percent"] >= own_range[0])
    & (players_df["selected_by_percent"] <= own_range[1])
]
if position_filter:
    filtered = filtered[filtered["position"].isin(position_filter)]
if team_filter:
    filtered = filtered[filtered["team_name"].isin(team_filter)]

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("FPL Agent Dashboard")
st.caption(f"{len(filtered)} players shown · {completed_gameweeks} gameweek(s) completed · "
           f"value basis: {value_mode_label}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_squad, tab_scatter, tab_watch, tab_smart, tab_value, tab_watchlist, tab_fixtures = st.tabs(
    ["My Squad", "Value Scatter", "Pre-season Watch", "Smart Picks", "Top Value",
     "Watchlist", "Fixtures"]
)

with tab_squad:
    if squad_df is not None:
        flagged = squad_df[squad_df["status"] != "a"]
        for _, p in flagged.iterrows():
            label = STATUS_LABELS.get(p["status"], p["status"])
            msg = f"**{p['web_name']}** - {label}"
            if p["news"]:
                msg += f": {p['news']}"
            st.warning(msg)

        st.dataframe(
            squad_df[["web_name", "position", "team_name", "price", "value_basis",
                      "selected_by_percent", "status", "role"]],
            width="stretch", hide_index=True,
        )
        suggestions = suggest_transfers(squad_df, players_df, bank, fixture_lookup)
        if not suggestions.empty:
            st.markdown(f"**Possible upgrades** (bank: £{bank}m, fixtures factored in)")
            st.dataframe(suggestions, width="stretch", hide_index=True)
        else:
            st.caption("No clear upgrades found within budget.")
    else:
        st.info("Switch to Live mode with your entry ID to see your squad here.")

with tab_scatter:
    view = st.radio("View", ["Price vs Ownership", "Price vs Points"], horizontal=True)
    if view == "Price vs Ownership":
        fig = px.scatter(
            filtered, x="price", y="selected_by_percent", color="position",
            color_discrete_map=POSITION_COLORS,
            hover_data=["web_name", "team_name", "value_basis", "reliability"],
            labels={"price": "Price (£m)", "selected_by_percent": "Selected by (%)"},
        )
        y_col = "selected_by_percent"
    else:
        y_label = {"blended": "Value (blended)", "last_season": "Last season's points",
                   "this_season": "This season's points"}[value_mode]
        fig = px.scatter(
            filtered.dropna(subset=["value_basis"]), x="price", y="value_basis", color="position",
            color_discrete_map=POSITION_COLORS,
            hover_data=["web_name", "team_name", "selected_by_percent", "reliability"],
            labels={"price": "Price (£m)", "value_basis": y_label},
        )
        y_col = "value_basis"
    if squad_names:
        owned = filtered[filtered["web_name"].isin(squad_names)]
        fig.add_scatter(
            x=owned["price"], y=owned[y_col], mode="markers",
            marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
            name="Your squad", hoverinfo="skip",
        )
    show(style_scatter(fig))

with tab_watch:
    sort_mode = st.radio("Sort by", ["Score", "Position"], horizontal=True, key="watch_sort")
    choice = st.radio("Metric", ["Most Owned", "Price Risers", "Best Value", "Best Expected"], horizontal=True)
    if choice == "Most Owned":
        render_ranked_bar(filtered, "selected_by_percent", "Owned (%)", sort_mode)
    elif choice == "Price Risers":
        render_ranked_bar(filtered[filtered["cost_change_start"] > 0], "cost_change_start",
                           "Price change (£m)", sort_mode)
    elif choice == "Best Value":
        render_ranked_bar(filtered[~filtered["is_new"]], "value_per_million", "Points / £m", sort_mode)
    else:
        render_ranked_bar(filtered.dropna(subset=["ep_next"]), "ep_next", "Expected pts", sort_mode)

with tab_smart:
    st.caption("Ranked within position, so a GKP is only compared to other GKPs.")
    sort_mode = st.radio("Sort by", ["Score", "Position"], horizontal=True, key="smart_sort")
    pick = st.radio("View", ["Overpriced", "Underpriced", "Overowned", "Underowned"], horizontal=True)
    scored = add_percentiles(filtered)
    score_col = {"Overpriced": "overpriced_score", "Underpriced": "underpriced_score",
                 "Overowned": "overowned_score", "Underowned": "underowned_score"}[pick]
    render_ranked_bar(scored.dropna(subset=[score_col]), score_col, "Score (higher = more so)", sort_mode)

with tab_value:
    sort_mode = st.radio("Sort by", ["Score", "Position"], horizontal=True, key="value_sort")
    render_ranked_bar(top_value_picks(filtered, n=15), "value_per_million", "Points / £m", sort_mode, n=15)

with tab_watchlist:
    st.markdown("**New / unmatched players** - no last-season record, ranked by ownership")
    new_players = players_df[players_df["is_new"]].sort_values("selected_by_percent", ascending=False).head(12)
    if new_players.empty:
        st.caption("No unmatched players currently in the pool.")
    else:
        fig = px.bar(new_players.sort_values("selected_by_percent"), x="selected_by_percent", y="web_name",
                     color="position", color_discrete_map=POSITION_COLORS, orientation="h",
                     labels={"selected_by_percent": "Owned (%)", "web_name": ""})
        show(style_bar(fig, len(new_players)))

    st.markdown("**Trending**")
    trend_pick = st.radio("Direction", ["Rising", "Falling"], horizontal=True)
    trending = filtered[filtered["trend"] == trend_pick].sort_values(
        "form", ascending=(trend_pick == "Falling")).head(10)
    if trending.empty:
        st.caption("Nothing meets this yet - check back once more gameweeks are in.")
    else:
        st.dataframe(trending[["web_name", "team_name", "position", "form", "points_per_game", "reliability"]],
                     width="stretch", hide_index=True)

    st.markdown("**Player history**")
    if data_mode == "Live":
        options = squad_names if squad_names else players_df.sort_values(
            "selected_by_percent", ascending=False)["web_name"].head(50).tolist()
        picked_name = st.selectbox("Player", options)
        row = players_df[players_df["web_name"] == picked_name]
        if not row.empty:
            try:
                hist = load_player_history(int(row.iloc[0]["id"]))
                if hist.empty:
                    st.caption("No gameweek history yet this season.")
                else:
                    fig_p = px.bar(hist, x="gameweek", y="points", labels={"gameweek": "GW", "points": "Points"})
                    show(style_bar(fig_p, 6))
                    fig_price = px.line(hist, x="gameweek", y="price",
                                         labels={"gameweek": "GW", "price": "Price (£m)"})
                    show(style_fig(fig_price, height=280, show_legend=False))
            except Exception as e:
                st.caption(f"Couldn't load history: {e}")
    else:
        st.caption("Switch to Live mode to look up individual player history.")

with tab_fixtures:
    n_gw = st.slider("Gameweeks ahead", 3, 8, 5)
    if data_mode == "Live":
        try:
            fixture_diff_n = fixture_difficulty_table(load_live_fixtures(), teams_map, current_event, n=n_gw)
        except Exception:
            fixture_diff_n = fixture_diff
    else:
        fixture_diff_n = fixture_diff
    if fixture_diff_n.empty:
        st.caption("No fixture data available.")
    else:
        fig = px.bar(fixture_diff_n, x="avg_difficulty", y="team_name", orientation="h",
                     color="avg_difficulty", color_continuous_scale=["#34d399", "#fbbf24", "#fb7185"],
                     labels={"avg_difficulty": f"Avg difficulty (next {n_gw} GWs)", "team_name": ""})
        fig.update_yaxes(categoryorder="array", categoryarray=fixture_diff_n["team_name"].tolist())
        fig.update_layout(coloraxis_showscale=False)
        show(style_bar(fig, len(fixture_diff_n)))
        st.caption("Lower = easier run of fixtures. Feeds into the transfer suggestions on My Squad.")

st.divider()
st.caption("Heuristic suggestions only - always sanity-check before making a transfer.")

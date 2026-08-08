"""
Gaffer - single-file FPL dashboard (deploy via Streamlit Community Cloud).
Everything in one file on purpose, so it's easy to upload/edit from a phone.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

st.set_page_config(page_title="Gaffer", page_icon="⚽", layout="wide")

APP_TITLE = "⚽ Gaffer"
APP_TAGLINE = "Squad decisions, backed by data."

POSITION_COLORS = {"GKP": "#a78bfa", "DEF": "#60a5fa", "MID": "#34d399", "FWD": "#fb7185"}
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
POSITION_ORDER = ["GKP", "DEF", "MID", "FWD"]
STATUS_LABELS = {"a": "Available", "d": "Doubtful", "i": "Injured", "s": "Suspended",
                  "u": "Unavailable", "n": "Not available"}
BASE_URL = "https://fantasy.premierleague.com/api"
LAST_SEASON_CSV = ("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
                    "master/data/2025-26/players_raw.csv")
FONT = "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
K_GAMEWEEKS = 8        # gameweeks until "this season" fully outweighs "last season" in Blended mode
LOW_SAMPLE_STARTS = 3   # fewer starts than this this season = flagged as a small sample

DEFINITIONS = {
    "blended": (f"Blends last season's per-gameweek rate with this season's actual rate, shifting "
                f"fully to this season after {K_GAMEWEEKS} gameweeks. Stops one big or bad gameweek "
                f"from swinging everything."),
    "last_season": "Last season's (2025/26) final points total. Ignores anything from this season.",
    "this_season": "This season's actual points so far. 0 for everyone before Gameweek 1, and noisy "
                   "in the first few weeks - a hat-trick in Gameweek 1 will look misleadingly huge.",
    "overpriced": "Price rank minus output rank, within position. High score = priced like a star, "
                  "not producing like one.",
    "underpriced": "Output rank minus price rank, within position. High score = producing more than "
                   "the price tag suggests.",
    "overowned": "Ownership rank minus output rank, within position. High score = lots of managers "
                 "own them, output doesn't fully back it up.",
    "underowned": "Output rank minus ownership rank, within position. High score = performing well "
                  "without the ownership to match - a differential.",
    "price_vs_ownership": "Every player plotted by current price against what share of managers own "
                          "them. Useful for spotting expensive players nobody trusts yet.",
    "price_vs_points": "Every player plotted by current price against their value basis (selected "
                       "below). The closer to the top-left, the better the bargain.",
}


# ---------------------------------------------------------------------------
# Chart styling helpers - applied to every chart so they look consistent
# ---------------------------------------------------------------------------
def style_fig(fig, height=440, show_legend=True):
    fig.update_layout(
        template="plotly_white",
        title=dict(text="", font=dict(family=FONT, size=15, color="#111827")),
        font=dict(family=FONT, size=12, color="#374151"),
        margin=dict(l=8, r=8, t=40, b=8),
        height=height,
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(size=11), title=dict(text="")),
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


def value_basis_picker(key):
    """Local (not sidebar-hidden) value-basis toggle, with its meaning
    shown immediately underneath so nobody has to guess what it means."""
    label = st.radio("Value basis", ["Blended", "Last season only", "This season only"],
                      index=0, horizontal=True, key=key)
    mode = {"Blended": "blended", "Last season only": "last_season",
            "This season only": "this_season"}[label]
    st.caption(DEFINITIONS[mode])
    return mode, label


# ---------------------------------------------------------------------------
# FPL API client
# ---------------------------------------------------------------------------
class FPLClient:
    def __init__(self, timeout=15):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "gaffer/0.4"})
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
# Last season lookup - points AND price, the pre-season / comparison baseline
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_last_season_data():
    """Best-effort fetch of last season's final points + price, keyed by
    player name. Returns {} on any failure so the app degrades gracefully."""
    try:
        df = pd.read_csv(LAST_SEASON_CSV)
        key = (df["first_name"].fillna("") + " " + df["second_name"].fillna("")).str.lower().str.strip()
        df = df.assign(key=key)
        points = df.groupby("key")["total_points"].max().to_dict()
        price = df.groupby("key")["now_cost"].max().to_dict()
        return {k: {"points": points[k], "price": price[k] / 10.0} for k in points}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Gameweek + fixture helpers
# ---------------------------------------------------------------------------
def gameweek_progress(events):
    completed = sum(1 for e in events if e.get("finished"))
    upcoming = next((e["id"] for e in events if not e.get("finished")), None)
    current_event = upcoming if upcoming is not None else (events[-1]["id"] if events else 1)
    return completed, current_event


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


def team_upcoming_fixtures(fixtures, teams_map, team_id, current_event, n=3):
    """The actual next N fixtures for one team - opponent, venue, difficulty."""
    rows = []
    for f in fixtures:
        ev = f.get("event")
        if ev is None or f.get("finished"):
            continue
        if f["team_h"] == team_id:
            rows.append({"event": ev, "opponent": teams_map.get(f["team_a"], "?"),
                         "venue": "Home", "difficulty": f["team_h_difficulty"]})
        elif f["team_a"] == team_id:
            rows.append({"event": ev, "opponent": teams_map.get(f["team_h"], "?"),
                         "venue": "Away", "difficulty": f["team_a_difficulty"]})
    return pd.DataFrame(rows).sort_values("event").head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Core player data
# ---------------------------------------------------------------------------
def players_dataframe(bootstrap):
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    df = pd.DataFrame(bootstrap["elements"])
    df["team_id"] = df["team"]
    df["team_name"] = df["team"].map(teams)
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["price"] = df["now_cost"] / 10.0
    df["form"] = pd.to_numeric(df["form"], errors="coerce")
    df["points_per_game"] = pd.to_numeric(df["points_per_game"], errors="coerce")
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce")
    df["ep_next"] = pd.to_numeric(df.get("ep_next"), errors="coerce")
    df["cost_change_start"] = df.get("cost_change_start", 0) / 10.0
    df["transfers_in"] = pd.to_numeric(df.get("transfers_in", 0), errors="coerce")
    df["transfers_out"] = pd.to_numeric(df.get("transfers_out", 0), errors="coerce")
    df["transfers_balance"] = df["transfers_in"] - df["transfers_out"]
    df["starts"] = pd.to_numeric(df.get("starts", 0), errors="coerce").fillna(0)
    df["news"] = df.get("news", "").fillna("")

    completed_gameweeks, current_event = gameweek_progress(bootstrap["events"])
    df["completed_gameweeks"] = completed_gameweeks
    df["current_event"] = current_event

    lookup = load_last_season_data()
    key = (df["first_name"].fillna("") + " " + df["second_name"].fillna("")).str.lower().str.strip()
    df["last_season_points"] = key.map(lambda k: lookup.get(k, {}).get("points"))
    df["last_season_price"] = key.map(lambda k: lookup.get(k, {}).get("price"))

    df["is_new"] = df["last_season_points"].isna() & (df["minutes"] == 0)
    df["reliability"] = np.where(df["starts"] < LOW_SAMPLE_STARTS, "Low sample", "OK")

    cols = ["id", "web_name", "team_id", "team_name", "position", "price", "total_points",
            "last_season_points", "last_season_price", "is_new", "completed_gameweeks",
            "current_event", "points_per_game", "form", "selected_by_percent", "reliability",
            "starts", "minutes", "status", "news", "ep_next", "cost_change_start",
            "transfers_in", "transfers_out", "transfers_balance"]
    return df[cols].reset_index(drop=True)


def apply_value_basis(df, mode="blended"):
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

    basis = basis.where(~df["is_new"])
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
        diff = fixture_lookup.get(row["team_name"], 3.0)
        return base - (diff - 3.0) * 3.0

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


def season_comparison_df(players_df):
    """Last season vs this season, side by side: price, points (prorated
    to a full season for a fair comparison), and value. Only meaningful
    for players who were around last season."""
    df = players_df[~players_df["is_new"]].copy()
    cgw = int(df["completed_gameweeks"].iloc[0]) if len(df) else 0
    df["this_season_pace"] = (df["total_points"] / cgw * 38).round(1) if cgw > 0 else np.nan
    df["value_then"] = (df["last_season_points"] / df["last_season_price"]).round(2)
    df["value_now"] = (df["this_season_pace"] / df["price"]).round(2)
    df["value_delta"] = df["value_now"] - df["value_then"]
    df["price_delta"] = (df["price"] - df["last_season_price"]).round(1)
    return df


# ---------------------------------------------------------------------------
# Synthetic demo data (works with no network access)
# ---------------------------------------------------------------------------
def _demo_teams():
    return ["Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
            "Chelsea", "Crystal Palace", "Everton", "Fulham", "Leeds",
            "Liverpool", "Man City", "Man Utd", "Newcastle", "Nott'm Forest",
            "Sunderland", "Spurs", "West Ham", "Wolves", "Burnley"]


def generate_demo_players_df(n_players=220, seed=42):
    rng = np.random.default_rng(seed)
    teams = _demo_teams()
    first = ["J.", "M.", "L.", "K.", "D.", "R.", "A.", "S.", "T.", "B."]
    surnames = ["Silva", "Costa", "Johnson", "Mbeki", "Rahman", "Novak", "Fischer",
                "Diallo", "Petrov", "Okafor", "Larsen", "Ricci", "Haaland", "Saka",
                "Foden", "Palmer", "Bruno", "Salah", "Watkins", "Isak"]

    positions = rng.choice(list(POSITION_MAP.values()), size=n_players, p=[0.12, 0.35, 0.35, 0.18])
    team_col = rng.choice(teams, size=n_players)
    team_ids = pd.Series(team_col).map({t: i + 1 for i, t in enumerate(teams)}).values
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
    transfers_in = np.clip((quality ** 2) * 800000 + rng.normal(0, 30000, n_players), 0, None).astype(int)
    transfers_out = np.clip((1 - quality) * 300000 + rng.normal(0, 20000, n_players), 0, None).astype(int)
    starts = np.clip((minutes / 75).round(), 0, 10).astype(int)
    is_new = rng.random(n_players) < 0.04
    status = np.where(rng.random(n_players) < 0.06, rng.choice(["i", "d", "s"], n_players), "a")
    news = np.where(status != "a", "Knock picked up in training - assessed ahead of next match.", "")
    last_season_price = np.clip(price - cost_change_start - rng.normal(0, 0.3, n_players), 3.9, 15.0).round(1)
    names = [f"{rng.choice(first)}{rng.choice(surnames)}" for _ in range(n_players)]

    df = pd.DataFrame({
        "id": range(1, n_players + 1), "web_name": names, "team_id": team_ids, "team_name": team_col,
        "position": positions, "price": price, "total_points": points,
        "last_season_points": np.where(is_new, np.nan,
                                        np.clip(points + rng.normal(0, 15, n_players), 0, None).round()),
        "last_season_price": np.where(is_new, np.nan, last_season_price),
        "points_per_game": points_per_game, "form": form, "selected_by_percent": ownership,
        "minutes": minutes, "starts": starts, "status": status, "news": news,
        "ep_next": ep_next, "cost_change_start": cost_change_start,
        "transfers_in": transfers_in, "transfers_out": transfers_out, "is_new": is_new,
    })
    df["transfers_balance"] = df["transfers_in"] - df["transfers_out"]
    df["completed_gameweeks"] = 10  # demo simulates mid-season
    df["current_event"] = 11
    df["reliability"] = np.where(df["starts"] < LOW_SAMPLE_STARTS, "Low sample", "OK")
    return df.reset_index(drop=True)


def generate_demo_fixture_difficulty():
    teams = _demo_teams()
    rng = np.random.default_rng(7)
    diff = rng.uniform(1.8, 4.2, len(teams)).round(2)
    return pd.DataFrame({"team_name": teams, "avg_difficulty": diff,
                          "fixtures_count": 5}).sort_values("avg_difficulty").reset_index(drop=True)


def generate_demo_team_fixtures(team_name):
    teams = [t for t in _demo_teams() if t != team_name]
    rng = np.random.default_rng(abs(hash(team_name)) % (2 ** 31))
    opponents = rng.choice(teams, 3, replace=False)
    venues = rng.choice(["Home", "Away"], 3)
    diffs = rng.integers(1, 6, 3)
    events = [12, 13, 14]
    return pd.DataFrame({"event": events, "opponent": opponents, "venue": venues, "difficulty": diffs})


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
def load_squad_picks(entry_id, gw):
    client = FPLClient()
    picks = client.entry_picks(entry_id, gw)
    entry_info = client.entry(entry_id)
    bank = entry_info.get("last_deadline_bank", 0) / 10.0
    return picks, bank


@st.cache_data(ttl=600)
def load_player_history(player_id):
    return player_history_df(FPLClient(), player_id)


# ---------------------------------------------------------------------------
# Sidebar - just setup, nothing you need mid-session lives here
# ---------------------------------------------------------------------------
st.sidebar.title(APP_TITLE)
st.sidebar.caption(APP_TAGLINE)
data_mode = st.sidebar.radio("Data source", ["Demo (offline)", "Live"], index=0)
entry_id = st.sidebar.text_input("Your FPL entry ID", value="501017")
gw = st.sidebar.number_input("Gameweek", min_value=1, max_value=38, value=1)

picks_raw, bank, squad_names = None, None, None
teams_map, current_event = {}, 1

if data_mode == "Demo (offline)":
    raw_players_df = load_demo_data()
    fixture_diff = generate_demo_fixture_difficulty()
    fixtures_raw = None
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
            fixtures_raw = None
            fixture_diff = pd.DataFrame()
            st.sidebar.warning(f"Couldn't load fixtures: {e}")
        if entry_id:
            try:
                picks_raw, bank = load_squad_picks(int(entry_id), int(gw))
                ids = [p["element"] for p in picks_raw["picks"]]
                squad_names = raw_players_df[raw_players_df["id"].isin(ids)]["web_name"].tolist()
            except Exception as e:
                st.sidebar.warning(f"Couldn't load squad picks yet: {e}")
    except Exception as e:
        st.error(f"Couldn't reach the live FPL API: {e}")
        st.stop()

fixture_lookup = dict(zip(fixture_diff["team_name"], fixture_diff["avg_difficulty"])) if len(fixture_diff) else {}
raw_players_df["fixture_difficulty"] = raw_players_df["team_name"].map(fixture_lookup).fillna(3.0)
completed_gameweeks = int(raw_players_df["completed_gameweeks"].iloc[0]) if len(raw_players_df) else 0

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
st.markdown(f"""
<div style="background: linear-gradient(90deg, #6d28d9 0%, #2563eb 35%, #059669 70%, #e11d48 100%);
            padding: 1.3rem 1.5rem; border-radius: 14px; margin-bottom: 0.9rem;">
  <div style="color:white; font-size:1.7rem; font-weight:800; letter-spacing:-0.02em;">{APP_TITLE}</div>
  <div style="color:rgba(255,255,255,0.88); font-size:0.92rem; margin-top:2px;">{APP_TAGLINE}</div>
</div>
""", unsafe_allow_html=True)
st.caption(f"{completed_gameweeks} gameweek(s) completed this season")

# ---------------------------------------------------------------------------
# Filters - always visible in the main body, not tucked in the sidebar
# ---------------------------------------------------------------------------
with st.expander("🔍 Filters", expanded=True):
    fc1, fc2 = st.columns(2)
    with fc1:
        position_filter = st.multiselect("Position", ["GKP", "DEF", "MID", "FWD"])
        price_range = st.slider("Price range (£m)", 3.5, 15.5, (3.5, 15.5), step=0.5)
    with fc2:
        team_filter = st.multiselect("Team", sorted(raw_players_df["team_name"].dropna().unique()))
        own_range = st.slider("Ownership % range", 0.0, 100.0, (0.0, 100.0))
    min_minutes = st.slider("Minimum minutes played", 0, 3400, 0, step=100)

filtered_raw = raw_players_df[
    (raw_players_df["minutes"] >= min_minutes)
    & (raw_players_df["selected_by_percent"] >= own_range[0])
    & (raw_players_df["selected_by_percent"] <= own_range[1])
    & (raw_players_df["price"] >= price_range[0])
    & (raw_players_df["price"] <= price_range[1])
]
if position_filter:
    filtered_raw = filtered_raw[filtered_raw["position"].isin(position_filter)]
if team_filter:
    filtered_raw = filtered_raw[filtered_raw["team_name"].isin(team_filter)]
st.caption(f"{len(filtered_raw)} players match your filters")

if filtered_raw.empty:
    st.warning("No players match your current filters. Try widening the price, ownership, "
               "or position ranges above.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_squad, tab_scatter, tab_smart, tab_value, tab_fixtures, tab_compare, tab_watchlist = st.tabs(
    ["My Squad", "Value Scatter", "Smart Picks", "Top Value", "Fixtures",
     "Season Compare", "Watchlist"]
)

with tab_squad:
    if picks_raw is not None:
        mode, mode_label = value_basis_picker("squad_value_mode")
        players_df_full = apply_value_basis(raw_players_df, mode)
        squad_df = squad_from_picks(picks_raw, players_df_full)

        flagged = squad_df[squad_df["status"] != "a"]
        for _, p in flagged.iterrows():
            label = STATUS_LABELS.get(p["status"], p["status"])
            msg = f"**{p['web_name']}** - {label}"
            if p["news"]:
                msg += f": {p['news']}"
            st.warning(msg)

        st.dataframe(
            squad_df[["web_name", "position", "team_name", "price", "value_basis",
                      "selected_by_percent", "status", "role"]].rename(
                columns={"value_basis": f"Value ({mode_label})", "selected_by_percent": "Owned %"}),
            width="stretch", hide_index=True,
        )
        suggestions = suggest_transfers(squad_df, players_df_full, bank, fixture_lookup)
        if not suggestions.empty:
            st.markdown(f"**Possible upgrades** (bank: £{bank}m, fixtures factored in)")
            st.dataframe(suggestions, width="stretch", hide_index=True)
        else:
            st.caption("No clear upgrades found within budget.")
    else:
        st.info("Switch to Live mode with your entry ID to see your squad here.")

with tab_scatter:
    mode, mode_label = value_basis_picker("scatter_value_mode")
    filtered = apply_value_basis(filtered_raw, mode)
    view = st.radio("View", ["Price vs Ownership", "Price vs Points"], horizontal=True)
    st.caption(DEFINITIONS["price_vs_ownership" if view == "Price vs Ownership" else "price_vs_points"])
    if view == "Price vs Ownership":
        fig = px.scatter(
            filtered, x="price", y="selected_by_percent", color="position",
            color_discrete_map=POSITION_COLORS,
            hover_data=["web_name", "team_name", "value_basis", "reliability"],
            labels={"price": "Price (£m)", "selected_by_percent": "Selected by (%)"},
        )
        y_col = "selected_by_percent"
    else:
        fig = px.scatter(
            filtered.dropna(subset=["value_basis"]), x="price", y="value_basis", color="position",
            color_discrete_map=POSITION_COLORS,
            hover_data=["web_name", "team_name", "selected_by_percent", "reliability"],
            labels={"price": "Price (£m)", "value_basis": f"Value ({mode_label})"},
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

    st.markdown("**Player detail**")
    detail_options = filtered.sort_values("selected_by_percent", ascending=False)["web_name"].tolist()
    if detail_options:
        picked = st.selectbox("Select a player", detail_options, key="scatter_detail_pick")
        row = filtered[filtered["web_name"] == picked].iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Team", row["team_name"])
        c2.metric("Position", row["position"])
        c3.metric("Price", f"£{row['price']}m")
        c4.metric("Points this season", int(row["total_points"]))
        if data_mode == "Live" and fixtures_raw:
            fx = team_upcoming_fixtures(fixtures_raw, teams_map, int(row["team_id"]), current_event, n=3)
        elif data_mode == "Demo (offline)":
            fx = generate_demo_team_fixtures(row["team_name"])
        else:
            fx = pd.DataFrame()
        if fx.empty:
            st.caption("No upcoming fixture data available.")
        else:
            st.caption("Next 3 fixtures (difficulty: 1 easy - 5 hard)")
            st.dataframe(fx, width="stretch", hide_index=True)

with tab_smart:
    mode, mode_label = value_basis_picker("smart_value_mode")
    filtered = apply_value_basis(filtered_raw, mode)
    st.caption("Ranked within position, so a GKP is only compared to other GKPs.")
    sort_mode = st.radio("Sort by", ["Score", "Position"], horizontal=True, key="smart_sort")
    pick = st.radio("View", ["Overpriced", "Underpriced", "Overowned", "Underowned"], horizontal=True)
    st.info(DEFINITIONS[pick.lower()])
    scored = add_percentiles(filtered)
    score_col = {"Overpriced": "overpriced_score", "Underpriced": "underpriced_score",
                 "Overowned": "overowned_score", "Underowned": "underowned_score"}[pick]
    render_ranked_bar(scored.dropna(subset=[score_col]), score_col, "Score (higher = more so)", sort_mode)

with tab_value:
    mode, mode_label = value_basis_picker("topvalue_value_mode")
    filtered = apply_value_basis(filtered_raw, mode)
    st.caption(f"Points per £m, using {mode_label.lower()} points. Best bargains in the pool right now.")
    sort_mode = st.radio("Sort by", ["Score", "Position"], horizontal=True, key="value_sort")
    render_ranked_bar(top_value_picks(filtered, n=15), "value_per_million", "Points / £m", sort_mode, n=15)

with tab_fixtures:
    st.caption("Team-level fixture difficulty (1 easy - 5 hard), and FPL's own points forecast "
               "for the next gameweek.")
    n_gw = st.slider("Gameweeks ahead", 3, 8, 5)
    if data_mode == "Live" and fixtures_raw:
        fixture_diff_n = fixture_difficulty_table(fixtures_raw, teams_map, current_event, n=n_gw)
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
        st.caption("Lower = easier run. Feeds into the transfer suggestions on My Squad.")

        st.markdown("**See a specific team's fixtures**")
        team_pick = st.selectbox("Team", fixture_diff_n["team_name"].tolist())
        if data_mode == "Live" and fixtures_raw:
            team_id = raw_players_df.loc[raw_players_df["team_name"] == team_pick, "team_id"]
            team_fx = team_upcoming_fixtures(fixtures_raw, teams_map, int(team_id.iloc[0]),
                                              current_event, n=n_gw) if len(team_id) else pd.DataFrame()
        else:
            team_fx = generate_demo_team_fixtures(team_pick)
        if not team_fx.empty:
            st.dataframe(team_fx, width="stretch", hide_index=True)

    st.divider()
    st.markdown("**Best expected points (FPL's own forecast, next GW)**")
    ep_pool = filtered_raw.dropna(subset=["ep_next"])
    if ep_pool.empty:
        st.caption("FPL hasn't published next-gameweek projections yet - check back closer to the deadline.")
    else:
        render_ranked_bar(ep_pool, "ep_next", "Expected pts", "Score", n=10)

with tab_compare:
    st.caption("Last season vs this season, side by side - price then vs now, points paced to a "
               "full season for a fair comparison, and value then vs now. Players new to the pool "
               "this season aren't included (nothing to compare against).")
    comp = season_comparison_df(filtered_raw)
    direction = st.radio("Show", ["Biggest value gainers", "Biggest value fallers"], horizontal=True)
    comp_sorted = comp.sort_values("value_delta", ascending=(direction == "Biggest value fallers")).head(10)
    if comp_sorted.empty:
        st.caption("Not enough data yet.")
    else:
        fig = px.bar(comp_sorted.sort_values("value_delta"), x="value_delta", y="web_name",
                     color="position", color_discrete_map=POSITION_COLORS, orientation="h",
                     labels={"value_delta": "Value change (pts/£m)", "web_name": ""})
        show(style_bar(fig, len(comp_sorted)))

    st.markdown("**Look up a player**")
    options = comp.sort_values("selected_by_percent", ascending=False)["web_name"].tolist()
    if options:
        picked = st.selectbox("Player", options, key="compare_pick")
        r = comp[comp["web_name"] == picked].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Price", f"£{r['price']}m", f"{r['price_delta']:+.1f}")
        c2.metric("Points pace", f"{r['this_season_pace']:.0f}" if pd.notna(r["this_season_pace"]) else "-",
                  f"{(r['this_season_pace'] - r['last_season_points']):+.0f}" if pd.notna(r["this_season_pace"]) else None)
        c3.metric("Value (pts/£m)", f"{r['value_now']:.1f}" if pd.notna(r["value_now"]) else "-",
                  f"{r['value_delta']:+.1f}" if pd.notna(r["value_delta"]) else None)
        st.caption(f"Last season: £{r['last_season_price']}m, {int(r['last_season_points'])} points.")

with tab_watchlist:
    st.markdown("**New / unmatched players** - no last-season record, ranked by ownership")
    st.caption("Excluded from value rankings elsewhere since there's no baseline to judge them against.")
    new_players = filtered_raw[filtered_raw["is_new"]].sort_values("selected_by_percent", ascending=False).head(12)
    if new_players.empty:
        st.caption("No unmatched players currently in the pool.")
    else:
        fig = px.bar(new_players.sort_values("selected_by_percent"), x="selected_by_percent", y="web_name",
                     color="position", color_discrete_map=POSITION_COLORS, orientation="h",
                     labels={"selected_by_percent": "Owned (%)", "web_name": ""})
        show(style_bar(fig, len(new_players)))

    st.divider()
    st.markdown("**Price movers**")
    st.caption("Players whose price has moved most since the season started.")
    movers = filtered_raw[filtered_raw["cost_change_start"] != 0]
    if movers.empty:
        st.caption("No price movement recorded yet.")
    else:
        render_ranked_bar(movers[movers["cost_change_start"] > 0], "cost_change_start",
                          "Price change (£m)", "Score", n=8)

    st.divider()
    st.markdown("**Player history this season**")
    if data_mode == "Live":
        options = squad_names if squad_names else raw_players_df.sort_values(
            "selected_by_percent", ascending=False)["web_name"].head(50).tolist()
        picked_name = st.selectbox("Player", options, key="history_pick")
        row = raw_players_df[raw_players_df["web_name"] == picked_name]
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

st.divider()
st.caption("Heuristic suggestions only - always sanity-check before making a transfer.")

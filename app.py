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

POSITION_COLORS = {"GKP": "#a78bfa", "DEF": "#60a5fa", "MID": "#34d399", "FWD": "#fb7185"}
POSITION_MAP = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
BASE_URL = "https://fantasy.premierleague.com/api"
LAST_SEASON_CSV = ("https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/"
                    "master/data/2025-26/players_raw.csv")
FONT = "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif"


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
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                    font=dict(size=11), title=None),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#e5e7eb"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f3f4f6", zeroline=False, linecolor="#e5e7eb")
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor="#e5e7eb")
    return fig


def style_bar(fig, n_items=10):
    fig.update_traces(marker_line_width=0)
    return style_fig(fig, height=90 + 34 * n_items, show_legend=False)


def style_scatter(fig, height=520):
    fig.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=0.5, color="white")))
    return style_fig(fig, height=height)


# ---------------------------------------------------------------------------
# FPL API client
# ---------------------------------------------------------------------------
class FPLClient:
    def __init__(self, timeout=15):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "fpl-agent/0.2"})
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
# Last season lookup - the "value" baseline before this season has points
# ---------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def load_last_season_lookup():
    """Best-effort fetch of last season's final points, keyed by player name.
    Returns {} on any failure so the rest of the app degrades gracefully
    rather than crashing."""
    try:
        df = pd.read_csv(LAST_SEASON_CSV)
        key = (df["first_name"].fillna("") + " " + df["second_name"].fillna("")).str.lower().str.strip()
        return df.assign(key=key).groupby("key")["total_points"].max().to_dict()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def players_dataframe(bootstrap):
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    df = pd.DataFrame(bootstrap["elements"])
    df["team_name"] = df["team"].map(teams)
    df["position"] = df["element_type"].map(POSITION_MAP)
    df["price"] = df["now_cost"] / 10.0
    df["form"] = pd.to_numeric(df["form"], errors="coerce")
    df["selected_by_percent"] = pd.to_numeric(df["selected_by_percent"], errors="coerce")
    df["ep_next"] = pd.to_numeric(df.get("ep_next"), errors="coerce")
    df["cost_change_start"] = df.get("cost_change_start", 0) / 10.0
    df["transfers_in"] = pd.to_numeric(df.get("transfers_in", 0), errors="coerce")
    df["transfers_out"] = pd.to_numeric(df.get("transfers_out", 0), errors="coerce")
    df["transfers_balance"] = df["transfers_in"] - df["transfers_out"]

    # Last season's points, matched by name - real signal for brand-new
    # players simply won't have a match, which is correct: we don't know
    # anything about them yet, so we flag them rather than guessing.
    lookup = load_last_season_lookup()
    key = (df["first_name"].fillna("") + " " + df["second_name"].fillna("")).str.lower().str.strip()
    df["last_season_points"] = key.map(lookup)

    season_started = bool(df["minutes"].sum() > 0)
    df["season_started"] = season_started
    df["value_basis"] = df["total_points"] if season_started else df["last_season_points"]
    df["is_new"] = df["value_basis"].isna()
    df["value_per_million"] = (df["value_basis"] / df["price"]).round(2)

    cols = ["id", "web_name", "team_name", "position", "price", "total_points",
            "last_season_points", "value_basis", "value_per_million", "is_new",
            "season_started", "points_per_game", "form", "selected_by_percent",
            "minutes", "status", "ep_next", "cost_change_start",
            "transfers_in", "transfers_out", "transfers_balance"]
    return df[cols].sort_values("value_basis", ascending=False, na_position="last").reset_index(drop=True)


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
    """Within-position percentile ranks, so a GKP isn't compared to a FWD
    on raw points. Used for the overpriced/overowned/underowned views."""
    df = df.copy()
    valid = ~df["is_new"]
    df["price_pct"] = np.nan
    df["value_pct"] = np.nan
    df["own_pct"] = np.nan
    if valid.any():
        df.loc[valid, "price_pct"] = df.loc[valid].groupby("position")["price"].rank(pct=True) * 100
        df.loc[valid, "value_pct"] = df.loc[valid].groupby("position")["value_basis"].rank(pct=True) * 100
        df.loc[valid, "own_pct"] = df.loc[valid].groupby("position")["selected_by_percent"].rank(pct=True) * 100
    df["overpriced_score"] = df["price_pct"] - df["value_pct"]
    df["overowned_score"] = df["own_pct"] - df["value_pct"]
    df["underowned_score"] = df["value_pct"] - df["own_pct"]
    return df


def suggest_transfers(squad_df, players_df, bank, n_suggestions=3):
    suggestions = []
    for _, player in squad_df.iterrows():
        if pd.isna(player.get("value_basis")):
            continue  # can't compare a player we have no baseline for
        budget = player["price"] + bank
        candidates = players_df[
            (players_df["position"] == player["position"])
            & (players_df["price"] <= budget)
            & (players_df["id"] != player["id"])
            & (players_df["status"] == "a")
            & (~players_df["is_new"])
        ]
        better = candidates[candidates["value_basis"] > player["value_basis"]]
        if not better.empty:
            best = better.sort_values("value_basis", ascending=False).iloc[0]
            suggestions.append({
                "out": player["web_name"], "out_pts": round(player["value_basis"], 0),
                "in": best["web_name"], "in_pts": round(best["value_basis"], 0),
                "in_price": best["price"],
                "cost_change": round(best["price"] - player["price"], 1),
            })
    if not suggestions:
        return pd.DataFrame()
    return pd.DataFrame(suggestions).sort_values("in_pts", ascending=False).head(n_suggestions)


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
    ep_next = np.clip(quality * 8 + rng.normal(0, 1, n_players), 0, 12).round(1)
    cost_change_start = np.round(
        np.where(rng.random(n_players) < 0.25,
                 (quality - 0.3) * rng.uniform(0.5, 2.0, n_players), 0.0), 1)
    transfers_in = np.clip((quality ** 2) * 800000 + rng.normal(0, 30000, n_players), 0, None).astype(int)
    transfers_out = np.clip((1 - quality) * 300000 + rng.normal(0, 20000, n_players), 0, None).astype(int)
    is_new = rng.random(n_players) < 0.04
    names = [f"{rng.choice(first)}{rng.choice(surnames)}" for _ in range(n_players)]

    df = pd.DataFrame({
        "id": range(1, n_players + 1), "web_name": names, "team_name": team_col,
        "position": positions, "price": price, "total_points": points,
        "last_season_points": np.where(is_new, np.nan,
                                        np.clip(points + rng.normal(0, 15, n_players), 0, None).round()),
        "points_per_game": (points / np.clip(minutes / 90, 1, None)).round(2),
        "form": form, "selected_by_percent": ownership, "minutes": minutes,
        "status": "a", "ep_next": ep_next, "cost_change_start": cost_change_start,
        "transfers_in": transfers_in, "transfers_out": transfers_out, "is_new": is_new,
    })
    df["transfers_balance"] = df["transfers_in"] - df["transfers_out"]
    df["season_started"] = True  # demo simulates mid-season so every chart has something to show
    df["value_basis"] = df["total_points"]
    df["value_per_million"] = (df["value_basis"] / df["price"]).round(2)
    return df.sort_values("value_basis", ascending=False).reset_index(drop=True)


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
# Sidebar: data source + filters
# ---------------------------------------------------------------------------
st.sidebar.title("⚽ FPL Agent")
data_mode = st.sidebar.radio("Data source", ["Demo (offline)", "Live"], index=0)
entry_id = st.sidebar.text_input("Your FPL entry ID", value="501017")
gw = st.sidebar.number_input("Gameweek", min_value=1, max_value=38, value=1)

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

season_started = bool(players_df["season_started"].iloc[0]) if len(players_df) else True

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
basis_note = ("this season's live points" if season_started else
              "last season's final points (2025/26), priced at this season's cost - "
              "flagged 'New' players have no match and are excluded from value rankings")
st.caption(f"{len(filtered)} players shown · value calculated from {basis_note}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_squad, tab_scatter, tab_watch, tab_smart, tab_value = st.tabs(
    ["My Squad", "Value Scatter", "Pre-season Watch", "Smart Picks", "Top Value"]
)

with tab_squad:
    if squad_df is not None:
        st.dataframe(
            squad_df[["web_name", "position", "team_name", "price", "value_basis",
                      "selected_by_percent", "role"]],
            width="stretch", hide_index=True,
        )
        suggestions = suggest_transfers(squad_df, players_df, bank)
        if not suggestions.empty:
            st.markdown(f"**Possible upgrades** (bank: £{bank}m)")
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
            hover_data=["web_name", "team_name", "value_basis"],
            labels={"price": "Price (£m)", "selected_by_percent": "Selected by (%)"},
        )
    else:
        y_label = "Points" if season_started else "Last season's points"
        fig = px.scatter(
            filtered.dropna(subset=["value_basis"]), x="price", y="value_basis", color="position",
            color_discrete_map=POSITION_COLORS,
            hover_data=["web_name", "team_name", "selected_by_percent"],
            labels={"price": "Price (£m)", "value_basis": y_label},
        )
    if squad_names:
        owned = filtered[filtered["web_name"].isin(squad_names)]
        y_col = "selected_by_percent" if view == "Price vs Ownership" else "value_basis"
        fig.add_scatter(
            x=owned["price"], y=owned[y_col], mode="markers",
            marker=dict(size=15, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
            name="Your squad", hoverinfo="skip",
        )
    st.plotly_chart(style_scatter(fig), width="stretch")

with tab_watch:
    choice = st.radio("Metric", ["Most Owned", "Price Risers", "Best Value", "Best Expected"],
                       horizontal=True)
    if choice == "Most Owned":
        d = filtered.sort_values("selected_by_percent", ascending=False).head(12)
        fig = px.bar(d.sort_values("selected_by_percent"), x="selected_by_percent", y="web_name",
                     color="position", color_discrete_map=POSITION_COLORS, orientation="h",
                     labels={"selected_by_percent": "Owned (%)", "web_name": ""})
    elif choice == "Price Risers":
        d = filtered[filtered["cost_change_start"] > 0].sort_values(
            "cost_change_start", ascending=False).head(12)
        if d.empty:
            st.caption("No price rises recorded yet.")
            d = None
        else:
            fig = px.bar(d.sort_values("cost_change_start"), x="cost_change_start", y="web_name",
                         color="position", color_discrete_map=POSITION_COLORS, orientation="h",
                         labels={"cost_change_start": "Price change (£m)", "web_name": ""})
    elif choice == "Best Value":
        d = filtered[~filtered["is_new"]].sort_values("value_per_million", ascending=False).head(12)
        fig = px.bar(d.sort_values("value_per_million"), x="value_per_million", y="web_name",
                     color="position", color_discrete_map=POSITION_COLORS, orientation="h",
                     labels={"value_per_million": "Points / £m", "web_name": ""})
    else:
        d = filtered.dropna(subset=["ep_next"]).sort_values("ep_next", ascending=False).head(12)
        if d.empty:
            st.caption("FPL hasn't published gameweek projections yet - check back closer to the deadline.")
            d = None
        else:
            fig = px.bar(d.sort_values("ep_next"), x="ep_next", y="web_name",
                         color="position", color_discrete_map=POSITION_COLORS, orientation="h",
                         labels={"ep_next": "Expected pts", "web_name": ""})
    if d is not None and not d.empty:
        st.plotly_chart(style_bar(fig, len(d)), width="stretch")

with tab_smart:
    st.caption("Ranked within position, so a GKP is only compared to other GKPs.")
    pick = st.radio("View", ["Overpriced", "Overowned", "Underowned"], horizontal=True)
    scored = add_percentiles(filtered)
    score_col = {"Overpriced": "overpriced_score", "Overowned": "overowned_score",
                 "Underowned": "underowned_score"}[pick]
    d = scored.dropna(subset=[score_col]).sort_values(score_col, ascending=False).head(12)
    if d.empty:
        st.caption("Not enough data yet for this view.")
    else:
        fig = px.bar(d.sort_values(score_col), x=score_col, y="web_name",
                     color="position", color_discrete_map=POSITION_COLORS, orientation="h",
                     hover_data=["price", "value_basis", "selected_by_percent"],
                     labels={score_col: "Score (higher = more so)", "web_name": ""})
        st.plotly_chart(style_bar(fig, len(d)), width="stretch")

with tab_value:
    d = top_value_picks(filtered, n=15)
    if d.empty:
        st.caption("No value data available for the current filters.")
    else:
        fig = px.bar(d.sort_values("value_per_million"), x="value_per_million", y="web_name",
                     color="position", color_discrete_map=POSITION_COLORS, orientation="h",
                     labels={"value_per_million": "Points / £m", "web_name": ""})
        st.plotly_chart(style_bar(fig, len(d)), width="stretch")

st.divider()
st.caption("Heuristic suggestions only - doesn't yet account for fixture runs, "
           "price-change timing, or remaining chips.")

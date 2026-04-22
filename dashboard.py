"""
dashboard.py — Streamlit dashboard for the Logistics Liability Tracker.

Reads from the local SQLite database ``liability_log.db`` and presents
KPIs, zone-level liability breakdowns, and a raw data log — suitable for
a capstone presentation demonstrating the business value of the AI
detection system.

Usage
-----
    streamlit run dashboard.py
    streamlit run dashboard.py -- --db path/to/liability_log.db
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ═════════════════════════════════════════════════════════════════════════
#  Config
# ═════════════════════════════════════════════════════════════════════════

DB_PATH = "liability_log.db"

# Allow overriding via CLI:  streamlit run dashboard.py -- --db other.db
if "--db" in sys.argv:
    idx = sys.argv.index("--db")
    if idx + 1 < len(sys.argv):
        DB_PATH = sys.argv[idx + 1]

# Severity → colour mapping (consistent with the tracker's palette)
STATUS_COLOURS = {
    "Normal":   "#22c55e",   # green
    "Minor":    "#eab308",   # yellow
    "Moderate": "#f97316",   # orange
    "Severe":   "#ef4444",   # red
}

ZONE_COLOURS = {
    "inbound":  "#3b82f6",   # blue
    "storage":  "#a855f7",   # purple
    "outbound": "#f59e0b",   # amber
}

ZONE_ORDER = ["inbound", "storage", "outbound"]


# ═════════════════════════════════════════════════════════════════════════
#  Data Layer
# ═════════════════════════════════════════════════════════════════════════

@st.cache_resource
def get_connection() -> sqlite3.Connection:
    """Open a persistent connection to the SQLite DB (shared across reruns)."""
    db = Path(DB_PATH)
    if not db.exists():
        st.error(f"Database not found: **{db.resolve()}**")
        st.info("Start the liability tracker first to create the database.")
        st.stop()
    return sqlite3.connect(str(db), check_same_thread=False)


def load_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """Fetch the full PackageHistory table as a DataFrame."""
    df = pd.read_sql_query(
        "SELECT PackageID, Zone, Timestamp, Final_Status FROM PackageHistory "
        "ORDER BY id DESC",
        conn,
    )
    if not df.empty:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


# ═════════════════════════════════════════════════════════════════════════
#  Page Config
# ═════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Logistics Liability Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS for a polished capstone-presentation look ─────────────
st.markdown("""
<style>
    /* ── Overall background ────────────────────────────────────── */
    .stApp {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    }

    /* ── Metric cards ──────────────────────────────────────────── */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* ── Section dividers ──────────────────────────────────────── */
    hr { border-color: rgba(148, 163, 184, 0.15) !important; }

    /* ── Header ────────────────────────────────────────────────── */
    .dashboard-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .dashboard-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #a855f7, #f59e0b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .dashboard-header p {
        color: #94a3b8;
        font-size: 1rem;
    }

    /* ── Chart containers ──────────────────────────────────────── */
    div[data-testid="stPlotlyChart"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 12px;
        padding: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════
#  Header
# ═════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="dashboard-header">
    <h1>📦 Logistics Liability Dashboard</h1>
    <p>AI-Powered Damage Detection  •  Zone Accountability  •  Live Data</p>
</div>
""", unsafe_allow_html=True)

# ── Refresh controls ─────────────────────────────────────────────────
col_refresh, col_auto = st.columns([1, 3])
with col_refresh:
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
with col_auto:
    auto_refresh = st.toggle("Auto-refresh (every 10 s)", value=False)

if auto_refresh:
    # Streamlit's st.rerun inside a timer fragment — polls the DB
    import time
    st.caption("⏱ Auto-refresh active — dashboard updates every 10 seconds")

st.divider()

# ═════════════════════════════════════════════════════════════════════════
#  Load Data
# ═════════════════════════════════════════════════════════════════════════

conn = get_connection()
df = load_data(conn)

if df.empty:
    st.warning("No data in the database yet. Run the liability tracker to start logging packages.")
    st.code("python logistics_liability_tracker.py --zone inbound", language="bash")
    st.stop()


# ═════════════════════════════════════════════════════════════════════════
#  KPI Metrics
# ═════════════════════════════════════════════════════════════════════════

total_packages = len(df)
damaged_packages = len(df[df["Final_Status"] != "Normal"])
damage_rate = (damaged_packages / total_packages * 100) if total_packages > 0 else 0.0
severe_count = len(df[df["Final_Status"] == "Severe"])

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Packages", f"{total_packages:,}")
k2.metric("Damaged Packages", f"{damaged_packages:,}")
k3.metric("Damage Rate", f"{damage_rate:.1f}%")
k4.metric("Severe Cases", f"{severe_count:,}")

st.divider()


# ═════════════════════════════════════════════════════════════════════════
#  Charts
# ═════════════════════════════════════════════════════════════════════════

chart_left, chart_right = st.columns(2)

# ── 1. Liability by Zone (core proof) ────────────────────────────────
with chart_left:
    st.subheader("🏭 Damage Count by Zone")
    st.caption("Which zone is causing the most damage? — the core liability metric")

    damaged_df = df[df["Final_Status"] != "Normal"]

    if not damaged_df.empty:
        zone_counts = (
            damaged_df.groupby("Zone")
            .size()
            .reindex(ZONE_ORDER, fill_value=0)
            .reset_index(name="Damaged Packages")
        )
        zone_counts.columns = ["Zone", "Damaged Packages"]

        fig_zone = px.bar(
            zone_counts,
            x="Zone",
            y="Damaged Packages",
            color="Zone",
            color_discrete_map=ZONE_COLOURS,
            text="Damaged Packages",
        )
        fig_zone.update_traces(
            textposition="outside",
            textfont_size=14,
            marker_line_width=0,
        )
        fig_zone.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            showlegend=False,
            xaxis=dict(title="", tickfont_size=13),
            yaxis=dict(title="Count", gridcolor="rgba(148,163,184,0.1)"),
            margin=dict(t=20, b=40, l=40, r=20),
            height=380,
        )
        st.plotly_chart(fig_zone, use_container_width=True)
    else:
        st.info("No damaged packages recorded yet.")


# ── 2. Status Distribution (pie / donut) ─────────────────────────────
with chart_right:
    st.subheader("📊 Status Distribution")
    st.caption("Breakdown of all scanned packages by final damage severity")

    status_counts = df["Final_Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]

    # Ensure consistent colour ordering
    colours = [STATUS_COLOURS.get(s, "#64748b") for s in status_counts["Status"]]

    fig_pie = go.Figure(go.Pie(
        labels=status_counts["Status"],
        values=status_counts["Count"],
        hole=0.45,
        marker=dict(colors=colours, line=dict(color="#1e293b", width=2)),
        textinfo="label+percent",
        textfont_size=13,
    ))
    fig_pie.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        showlegend=True,
        legend=dict(font_size=12),
        margin=dict(t=20, b=20, l=20, r=20),
        height=380,
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ── 3. Zone × Severity heatmap (stacked bar) ────────────────────────
st.subheader("🗺️ Zone × Severity Breakdown")
st.caption("Detailed view — which severity levels occur in each zone")

cross = (
    df.groupby(["Zone", "Final_Status"])
    .size()
    .reset_index(name="Count")
)

if not cross.empty:
    fig_stack = px.bar(
        cross,
        x="Zone",
        y="Count",
        color="Final_Status",
        color_discrete_map=STATUS_COLOURS,
        barmode="stack",
        category_orders={"Zone": ZONE_ORDER, "Final_Status": ["Normal", "Minor", "Moderate", "Severe"]},
        text="Count",
    )
    fig_stack.update_traces(textposition="inside", textfont_size=12)
    fig_stack.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend_title_text="Severity",
        xaxis=dict(title="", tickfont_size=13),
        yaxis=dict(title="Count", gridcolor="rgba(148,163,184,0.1)"),
        margin=dict(t=20, b=40, l=40, r=20),
        height=350,
    )
    st.plotly_chart(fig_stack, use_container_width=True)


# ── 4. Timeline — packages over time ────────────────────────────────
st.subheader("📈 Packages Over Time")

df_timeline = df.copy()
df_timeline["Date"] = df_timeline["Timestamp"].dt.date

timeline = (
    df_timeline.groupby(["Date", "Final_Status"])
    .size()
    .reset_index(name="Count")
)

if not timeline.empty:
    fig_time = px.area(
        timeline,
        x="Date",
        y="Count",
        color="Final_Status",
        color_discrete_map=STATUS_COLOURS,
        category_orders={"Final_Status": ["Normal", "Minor", "Moderate", "Severe"]},
    )
    fig_time.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend_title_text="Status",
        xaxis=dict(title="", gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(title="Count", gridcolor="rgba(148,163,184,0.1)"),
        margin=dict(t=20, b=40, l=40, r=20),
        height=300,
    )
    st.plotly_chart(fig_time, use_container_width=True)


st.divider()


# ═════════════════════════════════════════════════════════════════════════
#  Raw Data Table
# ═════════════════════════════════════════════════════════════════════════

st.subheader("🗄️ Package History Log")

# Filters
fcol1, fcol2, fcol3 = st.columns(3)
with fcol1:
    zone_filter = st.multiselect("Filter by Zone", options=ZONE_ORDER, default=ZONE_ORDER)
with fcol2:
    status_options = ["Normal", "Minor", "Moderate", "Severe"]
    status_filter = st.multiselect("Filter by Status", options=status_options, default=status_options)
with fcol3:
    search = st.text_input("🔍 Search Package ID", placeholder="e.g. PKG-VN-839172")

# Apply filters
filtered = df[
    (df["Zone"].isin(zone_filter)) &
    (df["Final_Status"].isin(status_filter))
]
if search:
    filtered = filtered[filtered["PackageID"].str.contains(search, case=False, na=False)]

st.caption(f"Showing {len(filtered):,} of {len(df):,} records")

# Colour-coded status column
def highlight_status(val):
    colour = STATUS_COLOURS.get(val, "#64748b")
    return f"color: {colour}; font-weight: 600;"

st.dataframe(
    filtered.style.applymap(highlight_status, subset=["Final_Status"]),
    use_container_width=True,
    height=400,
)


# ═════════════════════════════════════════════════════════════════════════
#  Auto-refresh mechanism
# ═════════════════════════════════════════════════════════════════════════

if auto_refresh:
    time.sleep(10)
    st.rerun()

"""
FedLineage  —  Federated MLOps Observability Dashboard
=======================================================
Streamlit prototype  |  BE Final Year Project
Pages: Overview, Client Health, Experiment Comparison, Model Lineage
Run:   streamlit run app.py
"""

import pathlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lineage as lineage_mod

# ── Constants ────────────────────────────────────────────────────────────────
EXPERIMENT_ID  = "FL_EXP_001"
DATASET        = "CIFAR-100"
ALGORITHM      = "FedAvg"
ANOMALY_ROUNDS = {6, 7}
STRAGGLER_ID   = "C3"
DATA_DIR       = pathlib.Path("data")

# ── Colour palette ───────────────────────────────────────────────────────────
ACCENT_BLUE   = "#6378FF"
ACCENT_TEAL   = "#00D9B5"
ACCENT_AMBER  = "#FFB347"
ACCENT_RED    = "#FF5C6C"
ACCENT_PURPLE = "#C46BFF"
GRID_COLOR    = "rgba(99,120,255,0.12)"
PAPER_BG      = "rgba(16,20,38,0.0)"

CLIENT_COLORS = {
    "C1": ACCENT_BLUE,
    "C2": ACCENT_TEAL,
    "C3": ACCENT_RED,
    "C4": ACCENT_PURPLE,
    "C5": ACCENT_AMBER,
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FedLineage | MLOps Dashboard",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family:'Inter',sans-serif; color:#E2E8F0; }
/* ── Dark canvas — apply to every Streamlit surface layer ── */
.stApp, [data-testid="stAppViewContainer"] { background:#0B1220 !important; }
[data-testid="stMain"], .stMain { background:#0B1220 !important; }
[data-testid="stMainBlockContainer"], .main .block-container { background:#0B1220 !important; }
body { background:#0B1220 !important; }
/* ── Streamlit chrome: style header dark, hide deploy noise ── */
#MainMenu{visibility:hidden;} footer{visibility:hidden;}
/* Paint the header bar to match the dark background */
[data-testid="stHeader"], .stAppHeader {
    background:#0B1220 !important;
    border-bottom:1px solid rgba(99,120,255,0.12);
}
/* Hide the Deploy button cleanly without breaking toolbar or sidebar controls */
[data-testid="stAppDeployButton"], .stAppDeployButton {
    display:none !important;
}
/* Ensure the sidebar collapse and expand buttons are always visible, functional, and clickable */
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    visibility:visible !important;
    opacity:1 !important;
    pointer-events:auto !important;
    display:inline-flex !important;
}
/* Tint the header and sidebar toggle buttons so they read clearly against the dark canvas */
[data-testid="stHeader"] button,
[data-testid="stSidebarHeader"] button,
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapseButton"] button {
    background:transparent !important;
    color:#8B9DC3 !important;
}
[data-testid="stHeader"] button svg,
[data-testid="stSidebarHeader"] button svg,
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapseButton"] svg {
    fill:#8B9DC3 !important;
    color:#8B9DC3 !important;
}
[data-testid="stHeader"] button:hover svg,
[data-testid="stSidebarHeader"] button:hover svg,
[data-testid="stExpandSidebarButton"]:hover svg,
[data-testid="stSidebarCollapseButton"]:hover svg {
    fill:#E2E8F0 !important;
    color:#E2E8F0 !important;
}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#0D1117;}
::-webkit-scrollbar-thumb{background:#2D3650;border-radius:3px;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#10142a 0%,#0D1117 100%);border-right:1px solid rgba(99,120,255,0.18);}
[data-testid="stSidebar"] *{color:#CBD5E1 !important;}
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1340px;}
.kpi-card{background:rgba(22,27,45,0.85);border:1px solid rgba(99,120,255,0.22);border-radius:14px;padding:1.1rem 1.3rem 1rem;text-align:left;backdrop-filter:blur(6px);height:100%;}
.kpi-card:hover{border-color:rgba(99,120,255,0.55);}
.kpi-label{font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#8B9DC3;margin-bottom:.35rem;}
.kpi-value{font-size:1.9rem;font-weight:700;line-height:1.15;color:#E2E8F0;font-family:'JetBrains Mono',monospace;}
.kpi-sub{font-size:.75rem;color:#8B9DC3;margin-top:.2rem;}
.kpi-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle;}
.section-title{font-size:.72rem;font-weight:600;letter-spacing:.12em;text-transform:uppercase;color:#8B9DC3;border-bottom:1px solid rgba(99,120,255,0.18);padding-bottom:.4rem;margin-bottom:.8rem;}
.alert-straggler{background:rgba(255,92,108,.12);border:1px solid rgba(255,92,108,.45);border-radius:10px;padding:.85rem 1.1rem;color:#FF8A96;font-size:.88rem;font-weight:500;margin-bottom:1rem;}
.info-box{background:rgba(0,217,181,.07);border:1px solid rgba(0,217,181,.25);border-radius:10px;padding:.9rem 1.1rem;font-size:.85rem;color:#A0F0E4;line-height:1.6;}
.client-card{background:rgba(22,27,45,.80);border:1px solid rgba(99,120,255,.18);border-radius:12px;padding:.9rem 1.1rem;margin-bottom:.6rem;}
.client-card.straggler{border-color:rgba(255,92,108,.5);background:rgba(255,92,108,.06);}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.7rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;}
.badge-healthy{background:rgba(0,217,181,.15);color:#00D9B5;border:1px solid rgba(0,217,181,.3);}
.badge-straggler{background:rgba(255,92,108,.15);color:#FF5C6C;border:1px solid rgba(255,92,108,.4);}
.proto-badge{background:rgba(196,107,255,.12);border:1px solid rgba(196,107,255,.3);border-radius:20px;padding:3px 12px;font-size:.68rem;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:#C46BFF;display:inline-block;}
.exp-table th{font-size:.72rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:#8B9DC3;padding:.5rem .8rem;}
.exp-table td{font-size:.88rem;padding:.55rem .8rem;border-top:1px solid rgba(99,120,255,.1);}
label{color:#8B9DC3 !important;font-size:.8rem !important;}
</style>
""", unsafe_allow_html=True)

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    clients     = pd.read_csv(DATA_DIR / "clients.csv")
    rounds      = pd.read_csv(DATA_DIR / "rounds.csv")
    experiments = pd.read_csv(DATA_DIR / "experiments.csv")
    return clients, rounds, experiments

try:
    clients_df, rounds_df, experiments_df = load_data()
except FileNotFoundError as exc:
    st.error(f"Data file not found: `{exc.filename}`. Run `python simulation.py` first.")
    st.stop()

LINEAGE_GRAPH = lineage_mod.build_lineage_graph(clients_df, rounds_df)
LINEAGE_ANOMALY = lineage_mod.find_anomaly(clients_df, rounds_df)

# ── Plotly helpers ────────────────────────────────────────────────────────────
def base_layout(**kw):
    d = dict(
        paper_bgcolor=PAPER_BG, plot_bgcolor=PAPER_BG,
        font=dict(family="Inter,sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=48,r=24,t=36,b=36),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(size=11), linecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR, tickfont=dict(size=11), linecolor=GRID_COLOR),
        hoverlabel=dict(bgcolor="#1A2035", bordercolor=ACCENT_BLUE, font_color="#E2E8F0", font_size=12),
        legend=dict(bgcolor="rgba(16,20,38,0.7)", bordercolor="rgba(99,120,255,0.25)", borderwidth=1, font=dict(size=11)),
    )
    d.update(kw)
    return d

def anomaly_vrect(fig):
    fig.add_vrect(x0=5.5, x1=7.5, fillcolor="rgba(255,92,108,0.08)", layer="below", line_width=0,
                  annotation_text="Straggler\nanomaly", annotation_position="top left",
                  annotation_font=dict(color="#FF8A96", size=10))
    return fig

def kpi_card(label, value, sub="", dot_color=None):
    dot = f'<span class="kpi-dot" style="background:{dot_color};"></span>' if dot_color else ""
    return f"""<div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{dot}{value}</div>
        <div class="kpi-sub">{sub}</div></div>"""

# ── Session-state: default page (only on a fresh session) ──────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "🏠  Overview"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:.8rem 0 1.4rem;">
        <div style="font-size:1.55rem;font-weight:700;letter-spacing:-.02em;color:#E2E8F0;">🔗 FedLineage</div>
        <div style="font-size:.78rem;color:#8B9DC3;margin-top:2px;line-height:1.4;">Federated MLOps Observability Platform</div>
        <div style="margin-top:.7rem;"><span class="proto-badge">Prototype · Simulated FL Env</span></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠  Overview", "💻  Client Health", "🧪  Experiment Comparison", "🧬  Model Lineage"],
        label_visibility="collapsed",
        key="page",
    )
    st.markdown("---")
    st.markdown('<div class="section-title">Experiment Context</div>', unsafe_allow_html=True)
    ctx = {"ID": EXPERIMENT_ID, "Dataset": DATASET, "Algorithm": ALGORITHM,
           "Clients": str(clients_df["client_id"].nunique()),
           "Rounds":  str(rounds_df["round"].max())}
    for k, v in ctx.items():
        st.markdown(f"""<div style="display:flex;justify-content:space-between;font-size:.8rem;padding:3px 0;color:#CBD5E1;">
            <span style="color:#8B9DC3;">{k}</span>
            <span style="font-family:'JetBrains Mono',monospace;font-weight:500;">{v}</span></div>""", unsafe_allow_html=True)
    st.markdown("""<br><div style="font-size:.68rem;color:#4A5A80;line-height:1.5;">
        FedLineage tracks performance together with execution metadata — not accuracy alone.<br><br>
        Model Lineage traces a version back to its contributing clients and round.</div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("""<h1 style="font-size:1.8rem;font-weight:700;margin-bottom:.1rem;letter-spacing:-.02em;color:#E2E8F0;">📊 Overview</h1>
    <p style="color:#8B9DC3;font-size:.87rem;margin-top:0;">Global model performance · Communication · Round-level telemetry</p>
    <hr style="border-color:rgba(99,120,255,.15);margin:.8rem 0 1.2rem;">""", unsafe_allow_html=True)

    num_clients    = clients_df["client_id"].nunique()
    num_rounds     = int(rounds_df["round"].max())
    final_acc      = rounds_df.loc[rounds_df["round"]==num_rounds,"global_accuracy"].values[0]
    total_comm     = rounds_df["communication_mb"].sum()
    straggler_cnt  = clients_df[clients_df["status"]=="straggler"]["client_id"].nunique()

    c1,c2,c3,c4,c5 = st.columns(5, gap="small")
    for col, lbl, val, sub, dot in [
        (c1,"Active Clients",    str(num_clients),           "FedAvg aggregation",    ACCENT_TEAL),
        (c2,"Training Rounds",   str(num_rounds),            "completed rounds",      ACCENT_BLUE),
        (c3,"Final Global Acc.", f"{final_acc:.1f}%",        f"round {num_rounds}",   ACCENT_TEAL),
        (c4,"Total Comm.",       f"{total_comm:.1f} MB",     "all rounds aggregate",  ACCENT_AMBER),
        (c5,"Stragglers",        str(straggler_cnt),         "anomalous clients",     ACCENT_RED),
    ]:
        col.markdown(kpi_card(lbl,val,sub,dot), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Accuracy | Loss
    col_a, col_l = st.columns(2, gap="medium")
    with col_a:
        st.markdown('<div class="section-title">Global Model Accuracy</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rounds_df["round"], y=rounds_df["global_accuracy"],
            mode="lines+markers", line=dict(color=ACCENT_TEAL,width=2.5),
            marker=dict(size=7,color=ACCENT_TEAL,line=dict(width=1.5,color="#0D1117")),
            hovertemplate="Round %{x}<br>Accuracy: %{y:.1f}%<extra></extra>"))
        dip = rounds_df.loc[rounds_df["global_accuracy"].idxmin()]
        fig.add_trace(go.Scatter(x=[dip["round"]], y=[dip["global_accuracy"]],
            mode="markers", marker=dict(size=13,color=ACCENT_RED,symbol="x",line=dict(width=2,color=ACCENT_RED)),
            hovertemplate="Dip — Round %{x}: %{y:.1f}%<extra></extra>", showlegend=False))
        anomaly_vrect(fig)
        fig.update_layout(**base_layout(yaxis_title="Accuracy (%)",xaxis_title="Round",height=300,showlegend=False,
            xaxis=dict(tickmode="linear",tick0=1,dtick=1,gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR),
            yaxis=dict(range=[60,92],gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR)))
        st.plotly_chart(fig, use_container_width=True)

    with col_l:
        st.markdown('<div class="section-title">Global Training Loss</div>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=rounds_df["round"], y=rounds_df["global_loss"],
            mode="lines+markers", line=dict(color=ACCENT_AMBER,width=2.5),
            marker=dict(size=7,color=ACCENT_AMBER,line=dict(width=1.5,color="#0D1117")),
            fill="tozeroy", fillcolor="rgba(255,179,71,0.07)",
            hovertemplate="Round %{x}<br>Loss: %{y:.3f}<extra></extra>"))
        spike = rounds_df.loc[rounds_df["global_loss"].idxmax()]
        fig2.add_trace(go.Scatter(x=[spike["round"]], y=[spike["global_loss"]],
            mode="markers", marker=dict(size=13,color=ACCENT_RED,symbol="x",line=dict(width=2,color=ACCENT_RED)),
            hovertemplate="Spike — Round %{x}: %{y:.3f}<extra></extra>", showlegend=False))
        anomaly_vrect(fig2)
        fig2.update_layout(**base_layout(yaxis_title="Loss",xaxis_title="Round",height=300,showlegend=False,
            xaxis=dict(tickmode="linear",tick0=1,dtick=1,gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR)))
        st.plotly_chart(fig2, use_container_width=True)

    # Comm overhead
    st.markdown('<div class="section-title">Communication Overhead by Round (MB)</div>', unsafe_allow_html=True)
    bar_cols = [ACCENT_RED if r in ANOMALY_ROUNDS else ACCENT_BLUE for r in rounds_df["round"]]
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=rounds_df["round"], y=rounds_df["communication_mb"],
        marker_color=bar_cols, marker_line=dict(width=0), opacity=0.85,
        hovertemplate="Round %{x}<br>Comm: %{y:.1f} MB<extra></extra>"))
    fig3.add_annotation(x=6.5, y=rounds_df["communication_mb"].max()+0.2,
        text="<b>Anomaly rounds</b>", showarrow=False, font=dict(color=ACCENT_RED,size=10))
    fig3.update_layout(**base_layout(yaxis_title="Communication (MB)",xaxis_title="Round",height=260,showlegend=False,
        xaxis=dict(tickmode="linear",tick0=1,dtick=1,gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR)))
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""<div class="info-box">
        <strong>🔍 About FedLineage</strong><br>
        Traditional FL systems report only final model accuracy. FedLineage augments this with
        execution metadata — per-client CPU/memory/bandwidth telemetry, training latency,
        communication overhead, and lineage provenance — so teams can diagnose <em>why</em>
        a model performed the way it did, not just <em>how well</em>.<br><br>
        The accuracy dip at <strong>Round 7</strong> is directly linked to client <strong>C3</strong>
        entering a straggler state (high CPU, low bandwidth, elevated loss).
        FedLineage surfaces this connection automatically.
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — CLIENT HEALTH
# ═════════════════════════════════════════════════════════════════════════════
elif page == "💻  Client Health":
    st.markdown("""<h1 style="font-size:1.8rem;font-weight:700;margin-bottom:.1rem;letter-spacing:-.02em;color:#E2E8F0;">💻 Client Health</h1>
    <p style="color:#8B9DC3;font-size:.87rem;margin-top:0;">Per-round client telemetry · Straggler detection · Resource utilisation</p>
    <hr style="border-color:rgba(99,120,255,.15);margin:.8rem 0 1.2rem;">""", unsafe_allow_html=True)

    available_rounds = sorted(clients_df["round"].unique().tolist())
    selected_round   = st.select_slider("Select FL Round", options=available_rounds, value=1,
                                        help="Inspect client telemetry for a specific FL round.")
    round_data = clients_df[clients_df["round"]==selected_round].copy()
    has_straggler = (round_data["status"]=="straggler").any()

    if has_straggler:
        sr = round_data[round_data["status"]=="straggler"].iloc[0]
        st.markdown(f"""<div class="alert-straggler">
            ⚠️ <strong>Potential Straggler Detected — Round {selected_round}</strong><br>
            Client <strong>{sr['client_id']}</strong> is exhibiting anomalous behaviour:
            CPU&nbsp;<strong>{sr['cpu_usage']}%</strong>,
            Memory&nbsp;<strong>{sr['memory_usage']}%</strong>,
            Bandwidth&nbsp;<strong>{sr['network_bandwidth']}&nbsp;MB/s</strong>,
            Training&nbsp;Time&nbsp;<strong>{sr['training_time']:.1f}s</strong> —
            significantly above peer baselines. This degrades global model convergence.
        </div>""", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.05, 0.95], gap="medium")

    with left_col:
        st.markdown(f'<div class="section-title">Client Telemetry — Round {selected_round}</div>', unsafe_allow_html=True)
        for _, row in round_data.sort_values("client_id").iterrows():
            is_str   = row["status"]=="straggler"
            card_cls = "client-card straggler" if is_str else "client-card"
            bdg_cls  = "badge badge-straggler"  if is_str else "badge badge-healthy"
            bdg_txt  = "straggler"              if is_str else "healthy"
            col_hex  = CLIENT_COLORS.get(row["client_id"], ACCENT_BLUE)
            def hl(val, bad_cond): return "#FF5C6C" if bad_cond else "#E2E8F0"
            st.markdown(f"""<div class="{card_cls}">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.55rem;">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span style="width:10px;height:10px;border-radius:50%;background:{col_hex};display:inline-block;"></span>
                        <strong style="font-size:1rem;color:#E2E8F0;">{row['client_id']}</strong>
                    </div>
                    <span class="{bdg_cls}">{bdg_txt}</span>
                </div>
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:.4rem .8rem;font-size:.8rem;">
                    <div><span style="color:#8B9DC3;">CPU</span><br>
                         <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:{hl(row['cpu_usage'],row['cpu_usage']>88)};">{row['cpu_usage']}%</span></div>
                    <div><span style="color:#8B9DC3;">Memory</span><br>
                         <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:{hl(row['memory_usage'],row['memory_usage']>88)};">{row['memory_usage']}%</span></div>
                    <div><span style="color:#8B9DC3;">Bandwidth</span><br>
                         <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:{hl(row['network_bandwidth'],row['network_bandwidth']<3)};">{row['network_bandwidth']} MB/s</span></div>
                    <div><span style="color:#8B9DC3;">Train Time</span><br>
                         <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:{hl(row['training_time'],row['training_time']>30)};">{row['training_time']:.1f}s</span></div>
                    <div><span style="color:#8B9DC3;">Local Loss</span><br>
                         <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:{hl(row['local_loss'],row['local_loss']>0.65)};">{row['local_loss']:.3f}</span></div>
                    <div><span style="color:#8B9DC3;">Local Acc.</span><br>
                         <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:{hl(row['local_accuracy'],row['local_accuracy']<73)};">{row['local_accuracy']:.1f}%</span></div>
                </div></div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown(f'<div class="section-title">Training Time by Client — Round {selected_round}</div>', unsafe_allow_html=True)
        sd = round_data.sort_values("client_id")
        bc = [CLIENT_COLORS.get(c, ACCENT_BLUE) for c in sd["client_id"]]
        fig_tt = go.Figure()
        fig_tt.add_trace(go.Bar(x=sd["client_id"], y=sd["training_time"],
            marker_color=bc, marker_line=dict(width=0), opacity=0.88,
            text=[f"{v:.1f}s" for v in sd["training_time"]], textposition="outside",
            textfont=dict(size=11,color="#E2E8F0"),
            hovertemplate="%{x}<br>Training Time: %{y:.1f}s<extra></extra>"))
        fig_tt.update_layout(**base_layout(yaxis_title="Training Time (s)",xaxis_title="Client",height=300,showlegend=False,
            xaxis=dict(gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR,range=[0,sd["training_time"].max()*1.22])))
        st.plotly_chart(fig_tt, use_container_width=True)

        st.markdown(f'<div class="section-title">Resource Radar — Round {selected_round}</div>', unsafe_allow_html=True)
        radar_cats = ["CPU %","Memory %","Bandwidth×10","TrainTime÷4","Loss×100"]
        fig_r = go.Figure()
        for _, row in sd.iterrows():
            vals = [row["cpu_usage"], row["memory_usage"],
                    min(100,row["network_bandwidth"]*10),
                    min(100,row["training_time"]/4*10),
                    min(100,row["local_loss"]*100)]
            cats  = radar_cats + [radar_cats[0]]
            vvals = vals + [vals[0]]
            cclr  = CLIENT_COLORS.get(row["client_id"], ACCENT_BLUE)
            fig_r.add_trace(go.Scatterpolar(r=vvals, theta=cats, fill="toself",
                name=row["client_id"], line=dict(color=cclr,width=1.8),
                fillcolor=cclr+"22", opacity=0.85))
        fig_r.update_layout(
            polar=dict(bgcolor="rgba(16,20,38,0)",
                radialaxis=dict(visible=True,range=[0,100],gridcolor=GRID_COLOR,
                    tickfont=dict(size=9,color="#8B9DC3"),linecolor=GRID_COLOR),
                angularaxis=dict(gridcolor=GRID_COLOR,tickfont=dict(size=10,color="#CBD5E1"),linecolor=GRID_COLOR)),
            paper_bgcolor=PAPER_BG, font=dict(family="Inter,sans-serif",color="#CBD5E1",size=11),
            margin=dict(l=30,r=30,t=30,b=30), height=310,
            legend=dict(bgcolor="rgba(16,20,38,0.7)",bordercolor="rgba(99,120,255,0.25)",borderwidth=1,font=dict(size=10)))
        st.plotly_chart(fig_r, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EXPERIMENT COMPARISON
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧪  Experiment Comparison":
    st.markdown("""<h1 style="font-size:1.8rem;font-weight:700;margin-bottom:.1rem;letter-spacing:-.02em;color:#E2E8F0;">🧪 Experiment Comparison</h1>
    <p style="color:#8B9DC3;font-size:.87rem;margin-top:0;">Multi-metric FL experiment analysis · Algorithm · Data distribution · Efficiency</p>
    <hr style="border-color:rgba(99,120,255,.15);margin:.8rem 0 1.2rem;">""", unsafe_allow_html=True)

    exp = experiments_df.copy()
    EXP_COLORS = {"EXP001":ACCENT_TEAL,"EXP002":ACCENT_RED,"EXP003":ACCENT_PURPLE}
    DIST_BADGE = {
        "IID":     ("rgba(0,217,181,.12)","#00D9B5","rgba(0,217,181,.3)"),
        "Non-IID": ("rgba(255,92,108,.12)","#FF8A96","rgba(255,92,108,.3)"),
    }

    st.markdown('<div class="section-title">Experiment Summary</div>', unsafe_allow_html=True)
    rows_html = ""
    max_acc = exp["final_accuracy"].max()
    for _, row in exp.iterrows():
        c = EXP_COLORS.get(row["experiment_id"], ACCENT_BLUE)
        db,dc,dbc = DIST_BADGE.get(row["data_distribution"],("rgba(99,120,255,.12)","#A0B0FF","rgba(99,120,255,.3)"))
        pct = row["final_accuracy"]/max_acc*100
        acc_bar = f"""<div style="display:flex;align-items:center;gap:8px;">
            <div style="flex:1;background:rgba(99,120,255,.12);border-radius:4px;height:6px;max-width:90px;">
                <div style="width:{pct:.0f}%;background:{c};height:100%;border-radius:4px;"></div></div>
            <span style="font-family:'JetBrains Mono',monospace;font-weight:600;color:{c};font-size:.85rem;">{row['final_accuracy']:.1f}%</span></div>"""
        strag = (f'<span style="color:#FF5C6C;font-weight:600;">{int(row["stragglers"])}</span>'
                 if row["stragglers"]>0 else '<span style="color:#00D9B5;">0</span>')
        rows_html += f"""<tr>
            <td><span style="width:10px;height:10px;border-radius:50%;background:{c};display:inline-block;margin-right:6px;vertical-align:middle;"></span>
                <strong style="font-family:'JetBrains Mono',monospace;font-size:.85rem;">{row['experiment_id']}</strong></td>
            <td><span style="font-size:.85rem;">{row['algorithm']}</span></td>
            <td><span style="background:{db};color:{dc};border:1px solid {dbc};border-radius:20px;padding:2px 10px;font-size:.7rem;font-weight:600;">{row['data_distribution']}</span></td>
            <td>{acc_bar}</td>
            <td><span style="font-family:'JetBrains Mono',monospace;font-size:.85rem;">{row['total_training_time']} min</span></td>
            <td><span style="font-family:'JetBrains Mono',monospace;font-size:.85rem;">{row['communication_mb']:.1f} MB</span></td>
            <td>{strag}</td></tr>"""

    st.markdown(f"""<div style="background:rgba(22,27,45,.7);border:1px solid rgba(99,120,255,.18);border-radius:12px;overflow:hidden;margin-bottom:1.2rem;">
    <table class="exp-table" style="width:100%;border-collapse:collapse;">
        <thead style="background:rgba(16,20,38,.6);">
            <tr><th>Experiment</th><th>Algorithm</th><th>Distribution</th><th>Final Accuracy</th>
                <th>Training Time</th><th>Communication</th><th>Stragglers</th></tr>
        </thead><tbody>{rows_html}</tbody></table></div>""", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2, gap="medium")
    clrs = [EXP_COLORS.get(e, ACCENT_BLUE) for e in exp["experiment_id"]]

    with ch1:
        st.markdown('<div class="section-title">Final Accuracy Comparison</div>', unsafe_allow_html=True)
        fa = go.Figure()
        fa.add_trace(go.Bar(x=exp["experiment_id"], y=exp["final_accuracy"],
            marker_color=clrs, marker_line=dict(width=0), opacity=0.88,
            text=[f"{v:.1f}%" for v in exp["final_accuracy"]], textposition="outside",
            textfont=dict(size=12,color="#E2E8F0"),
            hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>"))
        fa.update_layout(**base_layout(yaxis_title="Final Accuracy (%)",xaxis_title="Experiment",height=310,showlegend=False,
            xaxis=dict(gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR,range=[75,max_acc*1.1])))
        st.plotly_chart(fa, use_container_width=True)

    with ch2:
        st.markdown('<div class="section-title">Training Time Comparison (min)</div>', unsafe_allow_html=True)
        ft = go.Figure()
        ft.add_trace(go.Bar(x=exp["experiment_id"], y=exp["total_training_time"],
            marker_color=clrs, marker_line=dict(width=0), opacity=0.88,
            text=[f"{v} min" for v in exp["total_training_time"]], textposition="outside",
            textfont=dict(size=12,color="#E2E8F0"),
            hovertemplate="%{x}<br>Training Time: %{y} min<extra></extra>"))
        ft.update_layout(**base_layout(yaxis_title="Training Time (min)",xaxis_title="Experiment",height=310,showlegend=False,
            xaxis=dict(gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR),
            yaxis=dict(gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR,range=[100,exp["total_training_time"].max()*1.15])))
        st.plotly_chart(ft, use_container_width=True)

    st.markdown('<div class="section-title">Accuracy vs. Communication Overhead</div>', unsafe_allow_html=True)
    fs = go.Figure()
    for _, row in exp.iterrows():
        c = EXP_COLORS.get(row["experiment_id"], ACCENT_BLUE)
        fs.add_trace(go.Scatter(x=[row["communication_mb"]], y=[row["final_accuracy"]],
            mode="markers+text", name=row["experiment_id"],
            marker=dict(size=18+row["stragglers"]*8, color=c, opacity=0.85, line=dict(width=2,color="#0D1117")),
            text=[row["experiment_id"]], textposition="top center", textfont=dict(size=11,color=c),
            hovertemplate=(f"<b>{row['experiment_id']}</b><br>Algorithm: {row['algorithm']}<br>"
                           f"Distribution: {row['data_distribution']}<br>"
                           f"Accuracy: {row['final_accuracy']:.1f}%<br>"
                           f"Comm: {row['communication_mb']:.1f} MB<br>"
                           f"Stragglers: {int(row['stragglers'])}<extra></extra>")))
    fs.update_layout(**base_layout(xaxis_title="Communication Overhead (MB)",yaxis_title="Final Accuracy (%)",height=300,
        xaxis=dict(range=[exp["communication_mb"].min()-1,exp["communication_mb"].max()+1],gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR),
        yaxis=dict(range=[80,max_acc+2],gridcolor=GRID_COLOR,zerolinecolor=GRID_COLOR)))
    st.plotly_chart(fs, use_container_width=True)

    best_acc = exp.loc[exp["final_accuracy"].idxmax(),"experiment_id"]
    best_eff = exp.loc[exp["total_training_time"].idxmin(),"experiment_id"]
    fp_rows  = exp[exp["algorithm"]=="FedProx"]
    fa_rows  = exp[(exp["algorithm"]=="FedAvg") & (exp["data_distribution"]=="Non-IID")]
    fp_note  = ""
    if not fp_rows.empty and not fa_rows.empty:
        fp_note = (f"<br>· <strong>FedProx</strong> ({fp_rows.iloc[0]['experiment_id']}) outperforms FedAvg on "
                   f"Non-IID data ({fp_rows.iloc[0]['final_accuracy']:.1f}% vs {fa_rows.iloc[0]['final_accuracy']:.1f}%) "
                   f"— demonstrating algorithm-aware experiment tracking.")
    st.markdown(f"""<div class="info-box">
        <strong>📌 Key Takeaways</strong><br>
        · <strong>{best_acc}</strong> achieves the highest final accuracy ({exp['final_accuracy'].max():.1f}%).<br>
        · <strong>{best_eff}</strong> completes training in the shortest time ({exp['total_training_time'].min()} min).{fp_note}<br><br>
        FedLineage enables multi-metric experiment comparison — accuracy, efficiency,
        communication cost, and straggler impact — providing a holistic view that pure
        accuracy metrics alone cannot capture.
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL LINEAGE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧬  Model Lineage":
    st.markdown("""<h1 style="font-size:1.8rem;font-weight:700;margin-bottom:.1rem;letter-spacing:-.02em;color:#E2E8F0;">🧬 Model Lineage</h1>
    <p style="color:#8B9DC3;font-size:.87rem;margin-top:0;">Client → Update → Round → Global Model · version tracing · simulated root-cause view</p>
    <hr style="border-color:rgba(99,120,255,.15);margin:.8rem 0 1.2rem;">""", unsafe_allow_html=True)

    available_rounds = sorted(rounds_df["round"].astype(int).unique().tolist())
    default_round = LINEAGE_ANOMALY["round"] if LINEAGE_ANOMALY else available_rounds[-1]
    selected_round = st.select_slider(
        "Inspect global model version",
        options=available_rounds,
        value=default_round,
        format_func=lambda r: f"Global Model V{r}  (Round {r})",
        help="Each FL round produces a new global model version from participating client updates.",
    )

    rmeta = rounds_df.loc[rounds_df["round"] == selected_round].iloc[0]
    k1, k2, k3, k4 = st.columns(4, gap="small")
    k1.markdown(kpi_card("Model Version", f"V{selected_round}", "aggregated global model", ACCENT_TEAL), unsafe_allow_html=True)
    k2.markdown(kpi_card("Global Accuracy", f"{rmeta['global_accuracy']:.1f}%", f"round {selected_round}", ACCENT_BLUE), unsafe_allow_html=True)
    k3.markdown(kpi_card("Global Loss", f"{rmeta['global_loss']:.3f}", "after aggregation", ACCENT_AMBER), unsafe_allow_html=True)
    k4.markdown(kpi_card("Contributing Clients", str(int(rmeta["participating_clients"])), "updates in this round", ACCENT_PURPLE), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Lineage Graph — Global Model V{selected_round}</div>', unsafe_allow_html=True)
    fig_lin = lineage_mod.lineage_figure(LINEAGE_GRAPH, selected_round)
    st.plotly_chart(fig_lin, use_container_width=True)

    st.markdown("""<div class="info-box">
        Relationship shown: <strong>Client → Update → Round → Global Model</strong>.
        Previous global versions also feed the next round. Selecting a version reconstructs
        which client updates produced that model.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Model V{selected_round} Investigation</div>', unsafe_allow_html=True)

    round_clients = clients_df[clients_df["round"] == selected_round]
    stragglers = round_clients[round_clients["status"] == "straggler"]

    if not stragglers.empty:
        sr = stragglers.iloc[0]
        prev = rounds_df.loc[rounds_df["round"] == selected_round - 1]
        acc_note = "Model accuracy decreased." if (not prev.empty and rmeta["global_accuracy"] < prev.iloc[0]["global_accuracy"]) else "Associated client telemetry is anomalous."
        st.markdown(f"""<div class="alert-straggler">
            <strong>Model V{selected_round} Investigation</strong><br>
            {acc_note}<br><br>
            <strong>Associated anomaly:</strong><br>
            Client: <strong>{sr['client_id']}</strong><br>
            Round: <strong>{selected_round}</strong><br>
            CPU utilization: <strong>{sr['cpu_usage']}%</strong><br>
            Memory utilization: <strong>{sr['memory_usage']}%</strong><br>
            Network bandwidth: <strong>{sr['network_bandwidth']} MB/s</strong><br>
            Training time: <strong>{sr['training_time']:.1f} sec</strong><br><br>
            Status: <strong>Potential contributing straggler</strong>
        </div>""", unsafe_allow_html=True)
        st.markdown("""<div class="info-box">
            Lineage allows the operator to trace the affected model version back to the
            participating client and round. This is a <strong>simulated proof-of-concept</strong>;
            the system flags a potential contributing straggler from telemetry, and does not
            claim a causal algorithm.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="info-box">
            No straggler flag on <strong>Global Model V{selected_round}</strong>.
            All contributing client updates in this round are marked healthy.
            Select <strong>V{LINEAGE_ANOMALY['round'] if LINEAGE_ANOMALY else 7}</strong> to inspect the simulated anomaly path.
        </div>""", unsafe_allow_html=True)

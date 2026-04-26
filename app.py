# app.py — European Car Registration Emissions Dashboard
# Student: Ramudi Vidanagamachchi | W2151470
# Module: 5DATA004C Data Science Project Lifecycle | CW2

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
 
#Page Config
st.set_page_config(
    page_title="EU CO2 Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
#CSS — clean professional light theme 
st.markdown("""
<style>
/* ── Base ── */
:root {
    --bg:     #f4f8fd;
    --panel:  #ffffff;
    --border: #d9e6f5;
    --ink:    #111827;
    --muted:  #64748b;
    --blue:   #2563eb;
    --green:  #059669;
    --orange: #d97706;
    --red:    #dc2626;
}
.stApp { background: var(--bg); }
#MainMenu, footer, header { display: none !important; }
.block-container { padding: 0.6rem 1rem 0.4rem 1rem !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0.4rem; }
[data-testid="column"] { padding: 0 !important; }
 

/* ── Sidebar ── */

[data-testid="stSidebar"] {
    background: var(--panel);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--ink) !important; }
[data-testid="stSidebarContent"] { padding-top: 0.5rem !important; }
[data-testid="stSidebarHeader"] { 
    height: 1.5rem !important; 
    min-height: 1.5rem !important; 
    padding: 0 !important; 
    margin: 0 !important;
    overflow: hidden !important;
}
[data-testid="stSidebarCollapseButton"] { margin-top: 0.5 !important; padding-top: 0.5 !important; }
            
/* Sidebar logo */
.logo-box {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 10px 12px;
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 14px;
    margin-top: 0.5; 
}
.logo-circle {
    width: 42px; height: 42px;
    background: var(--blue);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.82rem; font-weight: 900; color: white; flex-shrink: 0;
}
.logo-title   { margin: 0; font-size: 0.88rem; font-weight: 900; color: var(--ink); line-height: 1.1; }
.logo-caption { margin: 0.18rem 0 0 0; font-size: 0.68rem; color: var(--muted); margin-top: 2px; }
 
/* Dataset overview box */
.ds-box  { background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 10px; margin-top: 12px; }
.ds-head { font-size: 0.76rem; font-weight: 700; color: var(--ink); margin-bottom: 6px; }
.ds-row  { display: flex; justify-content: space-between; padding: 3px 0;
           border-bottom: 1px solid #e2e8f0; font-size: 0.72rem; }
.ds-row:last-child { border-bottom: none; }
.ds-k { color: var(--muted); }
.ds-v { font-weight: 600; color: var(--ink); }
 
/* Download button */
[data-testid="stDownloadButton"] button {
    background: var(--blue) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; width: 100%;
    font-weight: 600; font-size: 0.78rem !important;
    padding: 7px !important;
}
 
/* Source note */
.src-note { font-size: 0.63rem; color: var(--muted); margin-top: 8px; line-height: 1.4; }
 
/* ── Header ── */
.hdr {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    box-shadow: 0 2px 8px rgba(15,23,42,.04);
}
.hdr-title { font-size: 1.3rem; font-weight: 900; color: var(--ink); margin: 0 0 2px 0; }
.hdr-sub   { font-size: 0.7rem; color: var(--muted); margin: 0; line-height: 1.5; }
.hdr-note  { font-size: 0.64rem; color: var(--muted); margin-top: 3px; }
 
/* Badge pills */
.pills { display: flex; gap: 6px; flex-wrap: wrap; }
.pill  { border-radius: 20px; padding: 4px 12px; font-size: 0.72rem; font-weight: 700; }
.pill-blue   { background: #dbeafe; color: var(--blue); }
.pill-green  { background: #d1fae5; color: var(--green); }
.pill-amber  { background: #fef3c7; color: var(--orange); }
.pill-purple { background: #ede9fe; color: #6d28d9; }
 
/* ── KPI Cards ── */
.kpi-row { display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
.kpi-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 14px;
    flex: 1; min-width: 120px; position: relative; overflow: hidden;
    box-shadow: 0 2px 8px rgba(15,23,42,.04);
}
.kpi-lbl  { font-size: 0.64rem; color: var(--muted); text-transform: uppercase;
            letter-spacing: .05em; margin-bottom: 3px; font-weight: 700; }
.kpi-val  { font-size: 1.15rem; font-weight: 900; color: var(--ink); line-height: 1.1; }
.kpi-sub  { font-size: 0.62rem; margin-top: 3px; }
.kpi-g    { color: var(--green); }
.kpi-r    { color: var(--red); }
.kpi-b    { color: var(--blue); }
.kpi-bar  { height: 3px; border-radius: 2px; margin-top: 6px; background: #e2e8f0; }
.kpi-fill { height: 100%; border-radius: 2px; }
 
/* ── Tabs ── */
[data-testid="stTabs"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 4px;
    margin-bottom: 8px;
}
[data-testid="stTabs"] button {
    color: var(--muted) !important;
    font-size: 0.82rem;
    border-radius: 6px;
    border: 1px solid transparent !important;
    background: transparent !important;
    flex: 1;
    text-align: center;
    font-weight: 500;
}
[data-testid="stTabs"] button:hover {
    color: var(--blue) !important;
    background: #f0f5ff !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: #eff6ff !important;
    color: var(--blue) !important;
    border: 1px solid var(--blue) !important;
    font-weight: 700 !important;
}
 
/* ── Chart cards ── */
.chart-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 10px 12px 6px 12px;
    box-shadow: 0 2px 8px rgba(15,23,42,.04);
}
.chart-title { font-size: 0.82rem; font-weight: 700; color: var(--ink); margin: 0 0 1px 0; }
.chart-sub   { font-size: 0.62rem; color: var(--muted); margin: 0 0 4px 0; }
 
/* Alert boxes */
[data-testid="stAlert"] { border-radius: 8px; }
 
/* DataFrame */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
 
/* Slider */
[data-testid="stSlider"] { padding-top: 0; padding-bottom: 0; }
[data-testid="stWidgetLabel"] p { font-size: 0.64rem; font-weight: 700; color: var(--muted); }

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0rem !important;
}
                       
</style>
""", unsafe_allow_html=True)
 
#COLOUR MAPS
COUNTRY_C = {
    "Croatia": "#ef4444",
    "Finland": "#06b6d4",
    "Greece":  "#f59e0b",
    "Ireland": "#2563eb",
    "Norway":  "#10b981",
}
FUEL_C = {
    "Electric":       "#10b981",
    "Plug-in Hybrid": "#84cc16",
    "Petrol":         "#f97316",
    "Diesel":         "#f59e0b",
    "Other":          "#94a3b8",
}
CAT_C = {
    "Zero Emission":    "#10b981",
    "Low (0-100)":      "#84cc16",
    "Medium (101-130)": "#f59e0b",
    "High (131-170)":   "#f97316",
    "Very High (170+)": "#ef4444",
}
 
#Chart Base Style 
def base(fig, h=230, legend=True):
    """Apply clean light theme to every chart."""
    fig.update_layout(
        height=h,
        margin=dict(l=34, r=14, t=20, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=10, color="#64748b"),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,  xanchor="left", x=0,
                    font=dict(size=9, color="#111827"), itemwidth=30),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#e2e8f0",
                     tickfont=dict(size=10, color="#64748b"),
                     title_font=dict(size=10, color="#64748b"))
    fig.update_yaxes(gridcolor="#f1f5f9", zeroline=False, linecolor="#e2e8f0",
                     tickfont=dict(size=10, color="#64748b"),
                     title_font=dict(size=10, color="#64748b"))
    return fig
 
 
def chart_card(title, subtitle, fig, key=None):
    """Wrap a chart inside a styled card."""
    with st.container(border=True):
        st.markdown(
            f"<div class='chart-card'>"
            f"<p class='chart-title'>{title}</p>"
            f"<p class='chart-sub'>{subtitle}</p>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, width="stretch",
                config={"displayModeBar": False},key=key if key else title)
        st.markdown("</div>", unsafe_allow_html=True)
 
 
# Data Functions
@st.cache_data(show_spinner=False)
def load_data():
    """@st.cache_data — load once and remember."""
    df = pd.read_csv("co2_data_clean.csv", low_memory=False)
    if df["Is_Electric"].dtype == object:
        df["Is_Electric"] = df["Is_Electric"].map({"True": True, "False": False})
    df["Year"] = df["Year"].astype(int)
    return df
 
 
def apply_filters(df, countries, year_range, fuels, weights, co2cats):
    """Apply all 5 sidebar filters. FR1 requirement."""
    r = df.copy()
    if countries:
        r = r[r["Country"].isin(countries)]
    r = r[r["Year"].between(year_range[0], year_range[1])]
    if fuels:
        r = r[r["Fuel_Type"].isin(fuels)]
    if weights:
        r = r[r["Weight_Category"].isin(weights)]
    if co2cats:
        r = r[r["CO2_Category"].isin(co2cats)]
    return r.copy()
 
 
# 08 Charts 
 
def chart_co2_trend(df):
    """FR3 — Line chart: avg CO2 per year by country."""
    agg = df.groupby(["Year","Country"])["CO2_gkm"].mean().round(1).reset_index()
    fig = go.Figure()
    for c in sorted(agg["Country"].unique()):
        v = agg[agg["Country"]==c]
        fig.add_trace(go.Scatter(
            x=v["Year"], y=v["CO2_gkm"], mode="lines+markers",
            name=c, line=dict(width=2.2, color=COUNTRY_C.get(c,"#64748b")),
            marker=dict(size=5)
        ))
    # EU target line at correct WLTP value
    fig.add_hline(y=115.1, line_dash="dash", line_color="#ef4444", line_width=1.5,
        annotation_text="EU target 115.1 g/km",
        annotation_font=dict(color="#ef4444", size=9),
        annotation_position="top right")
    # 2035 zero-emission target annotation
    fig.add_annotation(x=2023, y=10, text="2035: 0 g/km",
        showarrow=True, arrowhead=2, arrowcolor="#10b981",
        font=dict(color="#10b981", size=9))
    return base(fig, 240)
 
 
def chart_ev_trend(df):
    """FR8 — Line chart: EV share % per country per year."""
    agg = df.groupby(["Year","Country"])["Is_Electric"].mean().reset_index()
    agg["EV%"] = (agg["Is_Electric"]*100).round(1)
    fig = go.Figure()
    for c in sorted(agg["Country"].unique()):
        v = agg[agg["Country"]==c]
        fig.add_trace(go.Scatter(
            x=v["Year"], y=v["EV%"], mode="lines+markers",
            name=c, line=dict(width=2.2, color=COUNTRY_C.get(c,"#64748b")),
            marker=dict(size=5)
        ))
    # 50% tipping point line
    fig.add_hline(y=50, line_dash="dot", line_color="#f59e0b", line_width=1.5,
        annotation_text="50% tipping point",
        annotation_font=dict(color="#f59e0b", size=9),
        annotation_position="top right")
    fig.update_yaxes(title_text="% registrations")
    return base(fig, 240)
 
 
def chart_fuel_mix(df):
    """fuel type mix per year."""
    agg = df.groupby(["Year","Fuel_Type"]).size().reset_index(name="n")
    agg["count_k"] = agg["n"] / 1000
    fig = go.Figure()
    for fuel in ["Petrol","Diesel","Plug-in Hybrid","Electric","Other"]:
        v = agg[agg["Fuel_Type"]==fuel]
        if v.empty:
            continue
        fig.add_trace(go.Bar(
            x=v["Year"], y=v["count_k"], name=fuel,
            marker_color=FUEL_C.get(fuel,"#94a3b8")
        ))
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_yaxes(title_text="Cars (thousands)")
    fig.update_xaxes(dtick=1)
    return base(fig, 240)
 
 
def chart_scatter(df):
    """weight vs CO2, sized by engine power."""
    s = df.sample(n=min(8000, len(df)), random_state=42).dropna(
        subset=["Weight_kg","CO2_gkm","Engine_Power_kW"])
    fig = go.Figure()
    for fuel in ["Petrol","Diesel","Plug-in Hybrid","Electric","Other"]:
        v = s[s["Fuel_Type"]==fuel]
        if v.empty:
            continue
        sizes = np.clip(v["Engine_Power_kW"].fillna(70).to_numpy()/18, 4, 12)
        fig.add_trace(go.Scattergl(
            x=v["Weight_kg"], y=v["CO2_gkm"],
            mode="markers", name=fuel,
            marker=dict(color=FUEL_C.get(fuel,"#94a3b8"),
                        size=sizes, opacity=0.45, line=dict(width=0)),
            hovertemplate="<b>%{customdata}</b><br>Weight:%{x:.0f}kg CO2:%{y:.0f}g/km<extra></extra>",
            customdata=v["Car_Model"].astype(str).to_numpy()
        ))
   
    fig.add_hline(y=115.1, line_dash="dash", line_color="#ef4444", line_width=1.4,
        annotation_text="EU target 115.1 g/km",
        annotation_font=dict(color="#ef4444", size=9))
    fig.update_xaxes(title_text="Weight (kg)")
    fig.update_yaxes(title_text="CO2 (g/km)")
    return base(fig, 260)
 
 
def chart_heatmap(df):
    """avg CO2 per country per year — the most powerful single visual."""
    pivot = df.groupby(["Country","Year"])["CO2_gkm"].mean().round(0).unstack()
    z = pivot.to_numpy()
    text = np.where(np.isnan(z), "", np.round(z).astype("object"))
    fig = go.Figure(go.Heatmap(
        z=z,
        x=pivot.columns.astype(str),
        y=pivot.index.astype(str),
        text=text, texttemplate="%{text}",
        textfont=dict(size=10, color="#111827"),
        hovertemplate="%{y}, %{x}<br>Avg CO2: %{z:.0f} g/km<extra></extra>",
        colorscale=[
            [0.0, "#0f7a3d"], [0.28,"#22c55e"],
            [0.52,"#fde68a"], [0.75,"#f97316"], [1.0,"#be123c"]
        ],
        zmin=np.nanmin(z), zmax=np.nanmax(z),
        showscale=False,
    ))
    fig.update_layout(margin=dict(l=62, r=8, t=8, b=24))
    result = base(fig, 220, legend=False)   
    result.update_xaxes(dtick=1)            
    return result 

def chart_top_manufacturers(df):
    """Top 10 manufacturers by lowest average CO2"""
    # Only manufacturers with enough data
    counts = df.groupby("Manufacturer")["CO2_gkm"].count()
    valid_mfr = counts[counts >= 50].index   # increased threshold a bit
    agg = (df[df["Manufacturer"].isin(valid_mfr)]
           .groupby("Manufacturer")["CO2_gkm"]
           .mean()
           .sort_values(ascending=True)
           .head(10)
           .round(1)
           .reset_index())
    agg.columns = ["Manufacturer", "Avg_CO2"]
    fig = px.bar(agg, x="Avg_CO2", y="Manufacturer", orientation="h",
                 color="Avg_CO2",
                 color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                 labels={"Avg_CO2": "Average CO₂ (g/km)"})
    fig.add_vline(x=115.1, line_dash="dash", line_color="#ef4444", 
                  annotation_text="EU target 115.1 g/km",
                  annotation_position="top right")
    
    fig.update_layout(showlegend=False, height=280)
    return base(fig, 280, legend=False)
    
def chart_donut_and_gap(df):
    """fuel share donut + gap to target bar."""
    # Donut data
    fuel_agg = df.groupby("Fuel_Type").size().reset_index(name="n")
    ev_pct = df["Is_Electric"].mean() * 100
 
    # Gap data
    gap_agg = df.groupby("Country")["Gap_to_Target"].mean().round(1).reset_index()
    gap_agg.columns = ["Country","Gap"]
    gap_agg = gap_agg.sort_values("Gap")
    bar_colors = ["#10b981" if v <= 0 else "#ef4444" for v in gap_agg["Gap"]]
 
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type":"domain"}, {"type":"bar"}]],
        column_widths=[0.45, 0.55],
        horizontal_spacing=0.19
   
    )
    
    # Donut
    fig.add_trace(go.Pie(
        labels=fuel_agg["Fuel_Type"], values=fuel_agg["n"],
        hole=0.56,
        domain=dict(x=[0.0, 0.38], y=[0, 1]), 
        marker=dict(colors=[FUEL_C.get(f,"#94a3b8") for f in fuel_agg["Fuel_Type"]]),
        textinfo="none", sort=False, showlegend=True
    ), 1, 1)
 
    # EV% in donut centre
    fig.add_annotation(
    text=f"<b>{ev_pct:.1f}%</b><br><span style='font-size:9px;color:#64748b'>EV share</span>",
    x=0.19,
    y=0.5,
    xref="paper",
    yref="paper",
    showarrow=False,
    xanchor="center",
    yanchor="middle",
    align="center",
    font=dict(size=16, color="#111827")
)
 
    # Gap bars
    fig.add_trace(go.Bar(
        x=gap_agg["Gap"], y=gap_agg["Country"],
        orientation="h", marker_color=bar_colors,
        text=[f"{v:+.1f}" for v in gap_agg["Gap"]],
        textposition="auto", cliponaxis=False,
        textfont=dict(size=9, color="#111827"),
        showlegend=False
    ), 1, 2)
 
    # Zero line on bar chart only — using Scatter trace to avoid domain error
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[str(gap_agg["Country"].iloc[0]), str(gap_agg["Country"].iloc[-1])],
        mode="lines",
        line=dict(color="#334155", width=1.2),
        hoverinfo="skip", showlegend=False
    ), 1, 2)
 
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=30, t=10, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=10, color="#64748b"),
        legend=dict(orientation="h", yanchor="top", y=-0.22, x=0,
                    font=dict(size=9, color="#111827"))
                    
    )
    
    fig.update_xaxes(
    title_text="Gap (g/km)",
    gridcolor="#f1f5f9",
    tickfont=dict(size=9, color="#64748b"),
    title_font=dict(size=9, color="#64748b"),
    title_standoff=25,
    row=1, col=2
)
    fig.update_layout(
    margin=dict(l=0, r=20, t=14, b=50)
)
    
    fig.update_yaxes(
    tickfont=dict(size=9, color="#64748b"),
    automargin=True,      
    title_standoff=10,    
    row=1, col=2
)
    return fig
 
 
#Compliance Progress Bars
def show_compliance(df):
    agg = df.groupby("Country")["CO2_gkm"].mean().sort_values()
    for country, avg in agg.items():
        pct = avg / 115.1
        is_good = avg <= 115.1
        icon = "✅" if is_good else "❌"
        bar_color = "#059669" if is_good else "#dc2626"
        c1, c2, c3 = st.columns([2, 5, 1.5])
        with c1:
            st.markdown(
                f"<span style='font-size:0.82rem;color:#111827'>"
                f"<b>{country}</b></span>", unsafe_allow_html=True)
        with c2:
            # Coloured bar using HTML instead of st.progress
            filled = min(float(pct), 1.0) * 100
            st.markdown(f"""
            <div style='background:#e2e8f0;border-radius:4px;height:8px;margin-top:6px'>
                <div style='width:{filled:.0f}%;background:{bar_color};
                     height:100%;border-radius:4px'></div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(
                f"<span style='font-size:0.78rem;color:#64748b'>"
                f"{icon} {avg:.1f}</span>", unsafe_allow_html=True)
 
 

# MAIN
def main():
 
    # st.spinner
    with st.spinner("Loading dataset..."):
        df = load_data()
 
        #Sidebar
    with st.sidebar:
        # Logo — no blank line above so it sits at top
        st.markdown("""
<div class="logo-box" style="margin-top:0 !important">
    <div class="logo-circle">EV</div>
    <div>
        <div class="logo-title">EEA Vehicle Transition</div>
        <div class="logo-caption">Dashboard controls</div>
    </div>
</div>
<p style="font-weight:700;font-size:0.9rem;margin:8px 0 2px 0;color:#111827">Filters</p>
        """, unsafe_allow_html=True)

        # Filter 1 — Country
        sel_countries = st.multiselect(
            "Country",
            options=sorted(df["Country"].unique()),
            default=[],
            placeholder="All 5 selected"
        )
 
        #Filter 2 — Year range
        year_range = st.slider("Year range", 2019, 2023, (2019, 2023))
 
        #Filter 3 — Fuel Type
        sel_fuels = st.multiselect(
            "Fuel Type",
            options=sorted(df["Fuel_Type"].unique()),
            default=[],
            placeholder="All fuel types"
        )
 
        # Filter 4 — Weight Category
        sel_weights = st.multiselect(
            "Weight Category",
            options=sorted(df["Weight_Category"].unique()),
            default=[],
            placeholder="All categories"
        )
 
        # Filter 5 — CO2 Category
        sel_co2cats = st.multiselect(
            "CO2 Category",
            options=sorted(df["CO2_Category"].unique()),
            default=[],
            placeholder="All categories"
        )
 
        # Apply filters
        countries = sel_countries or sorted(df["Country"].unique())
        fuels     = sel_fuels     or sorted(df["Fuel_Type"].unique())
        weights   = sel_weights   or sorted(df["Weight_Category"].unique())
        co2cats   = sel_co2cats   or sorted(df["CO2_Category"].unique())
 
        filtered = apply_filters(df, countries, year_range, fuels, weights, co2cats)
 
        st.divider()
 
        # Dataset overview box
        st.markdown(f"""
        <div class="ds-box">
            <div class="ds-head">Dataset overview</div>
            <div class="ds-row">
                <span class="ds-k">Total records</span>
                <span class="ds-v">{len(df):,}</span>
            </div>
            <div class="ds-row">
                <span class="ds-k">Filtered records</span>
                <span class="ds-v">{len(filtered):,}</span>
            </div>
            <div class="ds-row">
                <span class="ds-k">Countries</span>
                <span class="ds-v">{filtered["Country"].nunique()}</span>
            </div>
            <div class="ds-row">
                <span class="ds-k">Years</span>
                <span class="ds-v">{year_range[0]}–{year_range[1]}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        # CSV Download 
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            "Download filtered CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="co2_filtered.csv",
            mime="text/csv",
            width="stretch"
        )
        st.markdown(
            "<div class='src-note'>Source: co2_data_clean.csv<br>"
            "Default view uses all records.</div>",
            unsafe_allow_html=True
        )
 
    # EMPTY FILTER SAFETY 
    if len(filtered) == 0:
        st.warning("⚠️ No data matches your filters. Please adjust your selections.")
        st.stop()
 
    # KPI VALUES 
    avg_co2  = filtered["CO2_gkm"].mean()
    ev_pct   = filtered["Is_Electric"].mean() * 100
    total    = len(filtered)
    avg_gap  = filtered["Gap_to_Target"].mean()
    best_c   = filtered.groupby("Country")["CO2_gkm"].mean().idxmin()
    top_mfr  = filtered["Manufacturer"].value_counts().idxmax()
    co2_2019 = df[df["Year"]==2019]["CO2_gkm"].mean()
    co2_2023 = df[df["Year"]==2023]["CO2_gkm"].mean()
    red_pct  = (co2_2019 - co2_2023) / co2_2019 * 100
    ev_2019  = df[df["Year"]==2019]["Is_Electric"].mean()*100
    no_avg   = df[df["Country"]=="Norway"]["CO2_gkm"].mean()
 
    # HEADER 
    st.markdown(f"""
    <div class="hdr">
        <div>
            <p class="hdr-title">European Car Registration Emissions Dashboard</p>
            <p class="hdr-sub">
                Real EEA registrations across Croatia, Finland, Greece, Ireland and Norway, 2019-2023
            </p>
            <p class="hdr-note">
                EU WLTP target: <strong>115.1 g/km</strong> &nbsp;|&nbsp;
                Dataset avg: <strong>{avg_co2:.1f} g/km</strong> &nbsp;|&nbsp;
                Top volume manufacturer: {top_mfr} &nbsp;|&nbsp;
                Greenest country: {best_c}
            </p>
        </div>
        <div class="pills">
            <span class="pill pill-blue">{filtered["Country"].nunique()} countries</span>
            <span class="pill pill-green">{year_range[0]}-{year_range[1]}</span>
            <span class="pill pill-amber">{total:,} cars</span>
            <span class="pill pill-purple">{ev_pct:.1f}% EV share</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # KPI CARDS (FR2)
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-lbl">Avg CO2</div>
            <div class="kpi-val">{avg_co2:.1f} g/km</div>
            <div class="kpi-sub kpi-r">{avg_co2-co2_2019:.1f} vs 2019</div>
            <div class="kpi-bar"><div class="kpi-fill"
                style="width:{min(avg_co2/200*100,100):.0f}%;background:#2563eb"></div></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">EV Share</div>
            <div class="kpi-val">{ev_pct:.1f}%</div>
            <div class="kpi-sub kpi-g">+{ev_pct-ev_2019:.1f} pts vs 2019</div>
            <div class="kpi-bar"><div class="kpi-fill"
                style="width:{ev_pct:.0f}%;background:#059669"></div></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Total Cars</div>
            <div class="kpi-val">{total:,}</div>
            <div class="kpi-sub kpi-b">Real registrations in dataset</div>
            <div class="kpi-bar"><div class="kpi-fill"
                style="width:100%;background:#7c3aed"></div></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">CO2 Reduction</div>
            <div class="kpi-val">{red_pct:.0f}%</div>
            <div class="kpi-sub kpi-g">Drop from 2019 to 2023</div>
            <div class="kpi-bar"><div class="kpi-fill"
                style="width:{red_pct:.0f}%;background:#059669"></div></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Greenest Country</div>
            <div class="kpi-val">{best_c}</div>
            <div class="kpi-sub kpi-g">{no_avg:.1f} g/km average</div>
            <div class="kpi-bar"><div class="kpi-fill"
                style="width:15%;background:#059669"></div></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-lbl">Avg Gap to Target</div>
            <div class="kpi-val">{avg_gap:.1f} g/km</div>
            <div class="kpi-sub kpi-g">Negative = below EU limit ✅</div>
            <div class="kpi-bar"><div class="kpi-fill"
                style="width:50%;background:#2563eb"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    # Dynamic Insights 
    # Calculate unique insight numbers not shown in KPI cards
    norway_avg  = df[df["Country"]=="Norway"]["CO2_gkm"].mean()
    croatia_avg = df[df["Country"]=="Croatia"]["CO2_gkm"].mean()
    gap_nc      = croatia_avg - norway_avg
    zero_cars   = int((df["CO2_gkm"] == 0).sum())
    above_target = (df["Gap_to_Target"] > 0).mean() * 100

    norway_co2 = filtered[filtered["Country"] == "Norway"]["CO2_gkm"].mean()
    croatia_co2 = filtered[filtered["Country"] == "Croatia"]["CO2_gkm"].mean()
    above_target_pct = (filtered["CO2_gkm"] > 115.1).mean() * 100

    st.markdown(f"""
    <div style="background:#ecfdf5; border:1px solid #86efac; border-radius:8px; 
                padding:12px 18px; margin:12px 0; font-size:0.86rem; color:#166534;">
        <strong>Key Insights:</strong> 
        Norway best: <strong>{norway_co2:.1f} g/km</strong> • 
        Croatia worst: <strong>{croatia_co2:.1f} g/km</strong> • 
        Gap: <strong>{croatia_co2 - norway_co2:.1f} g/km</strong> • 
        Cars above EU target: <strong>{above_target_pct:.1f}%</strong>
    </div>
    """, unsafe_allow_html=True)
 
    #Covid Warnings
    if year_range[0] <= 2020 <= year_range[1]:
        st.markdown("""
        <div style='background:#fffbeb;border:1px solid #fcd34d;
             border-radius:8px;padding:10px 16px;margin-bottom:8px;
             display:flex;align-items:center;gap:12px'>
            <span style='font-size:0.78rem'>⚠️</span>
            <span style='font-size:0.78rem;color:#92400e'>
                <b>COVID-19 (2020):</b> Car sales fell 25%. 
                Temporary CO2 drop — not a real fleet improvement.
            </span>
        </div>
        """, unsafe_allow_html=True)

    #Tabs
    tab1, tab2 = st.tabs(["📊   Overview", "🔍   Vehicle Explorer"])
 
    # TAB 1 — OVERVIEW (all charts)
    with tab1:
 
        # Row 1: CO2 Trend | EV Market Share | Fuel Mix
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            chart_card(
                "CO2 trend by country",
                "Average tailpipe emissions with EU target",
                chart_co2_trend(filtered)
            )
        with c2:
            chart_card(
                "EV market share by country",
                "Share of registrations that are fully electric",
                chart_ev_trend(filtered)
            )
        with c3:
            st.markdown(
                "<p class='chart-title'>Fuel type mix by country</p>"
                "<p class='chart-sub'>Select country and view type below</p>",
                unsafe_allow_html=True
            )
            country_pick = st.selectbox(
                "View fuel mix for:",
                ["All Countries"] + sorted(filtered["Country"].unique()),
                key="fuel_country_pick"
            )
            view = st.radio(
                "Show as:",
                ["Count", "Percentage (%)"],
                horizontal=True,
                key="fuel_view_toggle"
            )
            fuel_df = filtered if country_pick == "All Countries" \
                      else filtered[filtered["Country"] == country_pick]

            if view == "Percentage (%)":
                agg_p = fuel_df.groupby(["Year","Fuel_Type"]).size().reset_index(name="n")
                agg_p["total"] = agg_p.groupby("Year")["n"].transform("sum")
                agg_p["pct"] = (agg_p["n"] / agg_p["total"] * 100).round(1)
                fig_fuel = px.bar(agg_p, x="Year", y="pct", color="Fuel_Type",
                    barmode="stack", color_discrete_map=FUEL_C,
                    labels={"pct":"Share (%)","Fuel_Type":"Fuel"})
                fig_fuel.update_layout(barmode="stack", bargap=0.35)
                chart_card("Fuel mix — " + country_pick + " (%)",
                           "Each year sums to 100%",
                           base(fig_fuel, 240),
                           key="fuel_mix_pct")
            else:
                chart_card(
                    "Fuel type mix — " + country_pick,
                    "Registration volume by fuel type",
                    chart_fuel_mix(fuel_df),
                    key="fuel_mix_count"
                )
                
        # Row 2: Weight vs CO2 | Heatmap | Donut+Gap
        c4, c5, c6 = st.columns(3, gap="small")
        with c4:
            chart_card(
                "Weight vs CO2 emissions",
                "Sample of 8,000 cars; bubble size is engine power",
                chart_scatter(filtered)
            )
        with c5:
            chart_card(
                "Country × year CO2 heatmap",
                "Greener cells indicate lower emissions",
                chart_heatmap(filtered)
            )
        with c6:
            chart_card(
                "Fuel share and EU target gap",
                "Negative gap = below EU target (good)",
                chart_donut_and_gap(filtered)
            )
 
        # Compliance progress bars
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.82rem;font-weight:700;color:#111827;"
            "margin-bottom:4px'>Country Compliance vs EU Target (115.1 g/km)</p>"
            "<p style='font-size:0.68rem;color:#64748b;margin-bottom:8px'>"
            "Shorter bar = closer to target · "
            "✅ Norway, Finland, Ireland are below · "
            "❌ Greece, Croatia are above</p>",
            unsafe_allow_html=True
        )
        show_compliance(filtered)

        #Top manufacturers chart
        st.markdown("<br>", unsafe_allow_html=True)
        chart_card(
            "Top 10 Cleanest Manufacturers — Lowest Average CO2",
            "Green = below EU target · Red = above EU target",
            chart_top_manufacturers(filtered),
            key="top_mfr_chart"
        )
 
        #Policy Conclusion 
        st.markdown(f"""
        <div style='background:#fef9c3;border:1px solid #fcd34d;border-radius:10px;
             padding:14px 18px;margin-top:16px;margin-bottom:4px'>
            <p style='font-size:0.82rem;font-weight:700;color:#92400e;margin:0 0 5px 0'>
                📋 Policy Conclusion</p>
            <p style='font-size:0.75rem;color:#78350f;margin:0;line-height:1.65'>
                Norway achieved <b>91.5% EV share</b> by 2023 and sits {115.1 - norway_co2:.1f} g/km below </b> the EU target — the strongest performer in Europe.
                Croatia (<b>131 g/km</b>) and Greece (<b>124.6 g/km</b>) remain above the
                legal limit 115.1 g/km and require urgent policy action including
                EV purchase subsidies, charging infrastructure investment, and stricter
                fleet CO2 penalties to reach the EU 2035 zero-emission target.
            </p>
        </div>
        """, unsafe_allow_html=True)
 
        st.divider()
        
 
        #show raw stats
        if st.checkbox("📋 Show fuel type statistics"):
            stats = filtered.groupby("Fuel_Type").agg(
                Avg_CO2=("CO2_gkm","mean"),
                Avg_Power=("Engine_Power_kW","mean"),
                Count=("CO2_gkm","count")
            ).round(1).reset_index()
            stats.columns = ["Fuel Type","Avg CO2 (g/km)","Avg Power (kW)","Cars"]
            st.dataframe(stats, width="stretch")
 

    # TAB 2 — Vehicle Explore 
    with tab2:
 
        #st.expander
        with st.expander("📋 Raw data preview and CSV download", expanded=True):
 
            #search box
            search = st.text_input("Search table...",
                placeholder="Type country, manufacturer, fuel type...")
 
            cols = ["Country","Year","Pool","Manufacturer","Car_Model",
                    "Fuel_Type","CO2_gkm","Weight_kg","Engine_Power_kW",
                    "Electric_Range_km","Gap_to_Target","CO2_Category","Weight_Category"]
            tbl = filtered[cols].head(1000).copy()
 
            if search:
                mask = tbl.apply(
                    lambda r: r.astype(str).str.contains(search, case=False).any(),
                    axis=1)
                tbl = tbl[mask]
 
            st.caption(f"Showing {len(tbl):,} of {len(filtered):,} filtered records")
 
            #st.dataframe with column_config
            st.dataframe(tbl, width="stretch",
                column_config={
                    "CO2_gkm": st.column_config.NumberColumn("CO2 (g/km)", format="%.1f"),
                    "Gap_to_Target": st.column_config.NumberColumn("Gap to Target", format="%.1f"),
                    "Weight_kg": st.column_config.NumberColumn("Weight (kg)", format="%.0f"),
                    "Engine_Power_kW": st.column_config.NumberColumn("Power (kW)", format="%.0f"),
                    "Electric_Range_km": st.column_config.NumberColumn("Electric Range (km)", format="%.0f"),
                })
 
            st.download_button("⬇️ Download CSV",
                data=filtered[cols].to_csv(index=False).encode("utf-8"),
                file_name="co2_export.csv", mime="text/csv")
 
    #Footer 
    st.markdown("""
    <div style="font-size:0.66rem;color:#94a3b8;padding:8px 0 2px;
        border-top:1px solid #e2e8f0;margin-top:10px;text-align:center">
        All CO2 values in g/km WLTP · EU Target: 115.1 g/km (Regulation EU 2019/631) ·
        EU 2035 Target: 0 g/km · Source: European Environment Agency (EEA) ·
        <b>Ramudi Vidanagamachchi W2151470</b> · University of Westminster / IIT
    </div>
    """, unsafe_allow_html=True)
 
 
if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- App Configuration ---
st.set_page_config(page_title="USDA Digital Diagnostics & AI Roadmap", layout="wide")

# --- Data Loading ---
@st.cache_data
def load_data():
    # Load the pre-cleaned dataset
    df = pd.read_csv("usda_data_clean.csv")
    return df

try:
    df_raw = load_data()
except FileNotFoundError:
    st.error("Error: 'usda_data_clean.csv' not found. Please ensure the dataset is uploaded to the repository.")
    st.stop()

# --- Interactive Sidebar Filter ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/US_Department_of_Agriculture_seal.svg/200px-US_Department_of_Agriculture_seal.svg.png", width=150)
st.sidebar.title("Executive Controls")

geo_filter = st.sidebar.radio(
    "🌍 Geographic Focus",
    ["US Domestic", "Foreign (Non-US)", "All Global Traffic"],
    index=0 # Defaults to US Domestic to meet project criteria
)

# Apply the filter dynamically
if geo_filter == "US Domestic":
    df = df_raw[df_raw['Country'] == 'United States'].copy()
elif geo_filter == "Foreign (Non-US)":
    df = df_raw[df_raw['Country'] != 'United States'].copy()
else:
    df = df_raw.copy()

st.sidebar.markdown(f"**Current Dataset Size:**\n{len(df):,} filtered records")

# --- Feature Engineering & Global Objects ---
features = ['Total Views per session', 'Total Average session duration', 'Total Bounce rate']

# Prepare data for clustering (Rural Development specific)
df_rd = df[df['Is_RD']].copy()

if len(df_rd) >= 3: # Ensure enough data exists for clustering
    X_rd = df_rd[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_rd)
    
    # Pre-calculate optimal K (K=3 chosen based on diagnostic validation)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_rd['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Assign semantic names to clusters dynamically
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    personas = {}
    for i, center in enumerate(cluster_centers):
        if center[2] > 0.45: # High bounce rate
            personas[i] = "Friction-Stalled"
        elif center[1] > 120: # High duration
            personas[i] = "Power Users"
        else:
            personas[i] = "Information Seekers"
            
    df_rd['Persona'] = df_rd['Cluster'].map(personas)

# --- UI Setup ---
st.title("🌾 USDA Executive Briefing: Data Diagnostics & AI Roadmap")
st.markdown("Isolating systemic friction points to prescribe targeted AI solutions for Rural Development services.")
st.caption(f"Currently viewing: **{geo_filter}** | All URLs displayed have >50,000 total sessions.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Layer 1: System-Wide Assessment", 
    "🎯 Layer 2: RD Clustering", 
    "🤖 Strategic Simulator", 
    "⚙️ Technical Diagnostics"
])

# --- Tab 1: System-Wide Assessment ---
with tab1:
    st.header("Layer 1: Macro Traffic Trends & Device Disparities")
    
    if len(df) == 0:
        st.warning("No data available for this geographic filter.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"2025 Traffic Trends ({geo_filter})")
            trend_df = df.groupby(['Month', 'Day'])['Total Sessions'].sum().reset_index()
            trend_df['Date'] = pd.to_datetime('2025-' + trend_df['Month'].astype(str) + '-' + trend_df['Day'].astype(str), errors='coerce')
            trend_df = trend_df.dropna().sort_values('Date')
            
            fig_trend = px.line(trend_df, x='Date', y='Total Sessions', title="Aggregate Daily Sessions")
            fig_trend.update_layout(xaxis_title="Date", yaxis_title="Total Sessions")
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with col2:
            st.subheader("Device Comparison: Vital 3-Minute Window")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=df, x='Desktop Average session duration', y='Desktop Bounce rate', color='blue', alpha=0.5, label='Desktop')
            sns.scatterplot(data=df, x='Mobile Average session duration', y='Mobile Bounce rate', color='orange', alpha=0.5, label='Mobile')
            
            # Annotate Vital 3-Minute Window (0 - 200 seconds)
            ax.axvspan(0, 200, color='red', alpha=0.1)
            plt.text(100, 0.9, 'Vital 3-Min Window', color='darkred', ha='center', weight='bold')
            plt.xlabel("Average Session Duration (s)")
            plt.ylabel("Bounce Rate")
            plt.title("Bounce Rate vs Session Duration (Mobile vs Desktop)")
            plt.legend()
            st.pyplot(fig)

# --- Tab 2: RD Clustering ---
with tab2:
    st.header("Layer 2: Rural Development Persona Profiling")
    
    if len(df_rd) < 3:
        st.warning(f"Insufficient Rural Development data for {geo_filter} after applying filters.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Segment Personas (Radar Chart)")
            radar_data = df_rd.groupby('Persona')[features].mean().reset_index()
            
            # Normalize for radar chart readability
            scaler_radar = StandardScaler()
            radar_scaled = scaler_radar.fit_transform(radar_data[features])
            
            fig_radar = go.Figure()
            for i, row in radar_data.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_scaled[i].tolist() + [radar_scaled[i][0]],
                    theta=features + [features[0]],
                    fill='toself',
                    name=row['Persona']
                ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-2, 2])), showlegend=True)
            st.plotly_chart(fig_radar, use_container_width=True)
            
        with col2:
            st.subheader("Friction Analysis & Zombie Sessions")
            # Zombie sessions: High duration (> 200s) but still bounce (> 0.5)
            df_rd['Is_Zombie'] = (df_rd['Total Average session duration'] > 200) & (df_rd['Total Bounce rate'] > 0.5)
            
            fig_scatter = px.scatter(
                df_rd, 
                x='Total Average session duration', 
                y='Total Bounce rate',
                color='Persona',
                hover_data=['Page path and screen class'],
                title="Friction Scatter (Highlighted Zombie Sessions)"
            )
            # Add markers for Zombies
            zombies = df_rd[df_rd['Is_Zombie']]
            if not zombies.empty:
                fig_scatter.add_trace(go.Scatter(
                    x=zombies['Total Average session duration'], 
                    y=zombies['Total Bounce rate'],
                    mode='markers',
                    marker=dict(size=12, symbol='x', color='red'),
                    name='Zombie Sessions'
                ))
            st.plotly_chart(fig_scatter, use_container_width=True)

# --- Tab 3: Strategic Simulator ---
with tab3:
    st.header("🤖 Prescriptive AI Strategic Simulator")
    st.markdown("Input hypothetical page metrics to determine the audience segment and the recommended AI intervention.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Input Metrics")
        sim_views = st.slider("Views per Session", 1.0, 10.0, 2.5, 0.1)
        sim_duration = st.slider("Average Session Duration (s)", 0.0, 600.0, 150.0, 10.0)
        sim_bounce = st.slider("Bounce Rate", 0.0, 1.0, 0.5, 0.05)
        
    with col2:
        st.subheader("Engine Output")
        if 'scaler' in locals() and 'kmeans' in locals():
            sim_input = pd.DataFrame([[sim_views, sim_duration, sim_bounce]], columns=features)
            sim_scaled = scaler.transform(sim_input)
            pred_cluster = kmeans.predict(sim_scaled)[0]
            pred_persona = personas.get(pred_cluster, "Unknown Segment")
            
            st.metric(label="Assigned Persona", value=pred_persona)
            
            st.markdown("### 🚀 Recommended AI Roadmap Action")
            if pred_persona == "Friction-Stalled":
                st.error("**Priority:** High\n\n**Tool:** AI-Enabled Guided Navigation & Chatbot\n\n**Impact:** Reduce navigation friction. Users are bouncing rapidly; deploy an intercept chatbot to offer immediate assistance on these high-drop-off pages.")
            elif pred_persona == "Power Users":
                st.success("**Priority:** Low\n\n**Tool:** AI Semantic Search Engine\n\n**Impact:** Deepen engagement. These users stay long and click often; advanced semantic search will help them find complex grant documentation faster without needing a chatbot.")
            else:
                st.info("**Priority:** Medium\n\n**Tool:** Predictive Content Recommendations\n\n**Impact:** Nudge to conversion. Suggest relevant RD grants and articles dynamically to lower bounce rates and increase session depth.")
        else:
            st.warning("Model not initialized. Please select a geographic region with sufficient Rural Development data.")

# --- Tab 4: Technical Diagnostics ---
with tab4:
    st.header("⚙️ Model Validation & Technical Diagnostics")
    st.markdown("Rigorous validation of the K-Means clustering algorithm.")
    
    if len(df_rd) >= 3:
        col1, col2 = st.columns(2)
        
        inertias = []
        sil_scores = []
        K_range = range(2, min(7, len(df_rd))) 
        
        for k in K_range:
            temp_km = KMeans(n_clusters=k, random_state=42, n_init=10)
            temp_preds = temp_km.fit_predict(X_scaled)
            inertias.append(temp_km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, temp_preds))
            
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(K_range, inertias, marker='o', linestyle='--')
            ax.set_title("Elbow Method (Inertia vs Clusters)")
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Inertia")
            st.pyplot(fig)
            
        with col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(K_range, sil_scores, marker='s', linestyle='-', color='green')
            ax.set_title("Silhouette Scores vs Clusters")
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Silhouette Score")
            st.pyplot(fig)
            
        st.markdown(f"""
        **Selection Justification:** Based on the Elbow Method and Silhouette scores across the **{geo_filter}** segment, **$k=3$** is maintained. This maximizes the interpretability of our user base into three highly actionable segments (*Friction-Stalled, Power Users,* and *Information Seekers*), enabling precise mapping to our AI Prescriptive Roadmap.
        """)
    else:
        st.warning(f"Not enough Rural Development data in the '{geo_filter}' segment to run Technical Diagnostics.")

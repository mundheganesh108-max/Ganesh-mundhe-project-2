import streamlit as st
st.title("My Notebook App")
st.write("Hello from Jupyter!")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as _go
from sklearn.ensemble import IsolationForest

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="6G Smart Manufacturing - Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# DATA LOADING & PREPROCESSING
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_prep_data():
    # Load dataset
    df = pd.read_csv('Thales_Group_Manufacturing.csv', lineterminator='\n')
    
    # Clean string column whitespace/newlines
    df.columns = [c.strip() for c in df.columns]
    if 'Efficiency_Status' in df.columns:
        df['Efficiency_Status'] = df['Efficiency_Status'].astype(str).str.strip()
    
    # Combine Date and Timestamp into a single Datetime column
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
    df = df.sort_values(by=['Machine_ID', 'Datetime']).reset_index(drop=True)
    
    # Feature Engineering for Anomaly Detection
    feature_cols = [
        'Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 
        'Network_Latency_ms', 'Packet_Loss_%', 'Quality_Control_Defect_Rate_%', 
        'Production_Speed_units_per_hr', 'Predictive_Maintenance_Score', 'Error_Rate_%'
    ]
    
    # Fit Isolation Forest Model for Anomaly Scoring
    iso_model = IsolationForest(contamination=0.05, random_state=42)
    
    # Anomaly score: higher means more anomalous
    scores = -iso_model.fit_predict(df[feature_cols].fillna(0))
    # Normalize score between 0 and 100 for easy interpretation
    df['Anomaly_Score'] = (iso_model.decision_function(df[feature_cols].fillna(0)) * -100).round(2)
    # Min-Max Scaling to 0-100 range
    min_s, max_s = df['Anomaly_Score'].min(), df['Anomaly_Score'].max()
    df['Anomaly_Score'] = ((df['Anomaly_Score'] - min_s) / (max_s - min_s) * 100).round(2)

    return df

# Load data
try:
    df_raw = load_and_prep_data()
except Exception as e:
    st.error(f"Error loading dataset 'Thales_Group_Manufacturing.csv'. Please ensure the file is in the working directory. Details: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR / USER CAPABILITIES
# -----------------------------------------------------------------------------
st.sidebar.title("🎛️ Control Panel")

# Machine selector
all_machines = sorted(df_raw['Machine_ID'].unique())
selected_machines = st.sidebar.multiselect(
    "Select Machine ID(s):",
    options=all_machines,
    default=all_machines[:5] if len(all_machines) >= 5 else all_machines
)

# Operation mode filter
all_modes = df_raw['Operation_Mode'].dropna().unique().tolist()
selected_modes = st.sidebar.multiselect(
    "Operation Mode Filter:",
    options=all_modes,
    default=all_modes
)

# Risk threshold slider
high_risk_threshold = st.sidebar.slider(
    "High Risk Threshold (Anomaly Score):",
    min_value=50,
    max_value=95,
    value=70,
    step=5
)

medium_risk_threshold = st.sidebar.slider(
    "Medium Risk Threshold (Anomaly Score):",
    min_value=20,
    max_value=high_risk_threshold - 5,
    value=40,
    step=5
)

# Time window selector
min_date = df_raw['Datetime'].min().date() if pd.notnull(df_raw['Datetime'].min()) else None
max_date = df_raw['Datetime'].max().date() if pd.notnull(df_raw['Datetime'].max()) else None

if min_date and max_date:
    date_range = st.sidebar.date_input(
        "Select Time Window:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
else:
    date_range = None

# Filter data based on sidebar controls
df_filtered = df_raw.copy()

if selected_machines:
    df_filtered = df_filtered[df_filtered['Machine_ID'].isin(selected_machines)]

if selected_modes:
    df_filtered = df_filtered[df_filtered['Operation_Mode'].isin(selected_modes)]

if date_range and len(date_range) == 2:
    start_d, end_d = date_range
    df_filtered = df_filtered[
        (df_filtered['Datetime'].dt.date >= start_d) & 
        (df_filtered['Datetime'].dt.date <= end_d)
    ]

# Assign Maintenance Risk Levels
def categorize_risk(score):
    if score >= high_risk_threshold:
        return 'High Risk'
    elif score >= medium_risk_threshold:
        return 'Medium Risk'
    return 'Low Risk'

df_filtered['Risk_Level'] = df_filtered['Anomaly_Score'].apply(categorize_risk)

# -----------------------------------------------------------------------------
# DASHBOARD MODULES (TABS)
# -----------------------------------------------------------------------------
st.title("🏭 Predictive Maintenance & Anomaly Detection")
st.caption("6G-Integrated Smart Manufacturing Real-Time Monitoring Dashboard")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", 
    "📈 Anomaly Dashboard", 
    "🚨 Alert Panel", 
    "📜 Historical Risk Analysis"
])

# -----------------------------------------------------------------------------
# TAB 1: PREDICTIVE MAINTENANCE OVERVIEW
# -----------------------------------------------------------------------------
with tab1:
    st.header("Predictive Maintenance Overview")
    
    # KPI Metrics
    total_assets = df_filtered['Machine_ID'].nunique()
    high_risk_count = df_filtered[df_filtered['Risk_Level'] == 'High Risk']['Machine_ID'].nunique()
    med_risk_count = df_filtered[df_filtered['Risk_Level'] == 'Medium Risk']['Machine_ID'].nunique()
    avg_score = df_filtered['Anomaly_Score'].mean() if not df_filtered.empty else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monitored Assets", total_assets)
    col2.metric("High-Risk Assets", high_risk_count, delta_color="inverse")
    col3.metric("Medium-Risk Assets", med_risk_count, delta_color="inverse")
    col4.metric("Avg Fleet Anomaly Score", f"{avg_score:.1f}")

    st.markdown("---")

    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Risk Level Distribution Across Fleet")
        risk_counts = df_filtered['Risk_Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk_Level', 'Count']
        fig_pie = px.pie(
            risk_counts, 
            names='Risk_Level', 
            values='Count',
            color='Risk_Level',
            color_discrete_map={'High Risk': '#EF553B', 'Medium Risk': '#FECB52', 'Low Risk': '#00CC96'},
            hole=0.4
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("High-Risk Asset Breakdown")
        latest_status = (
            df_filtered.sort_values('Datetime')
            .groupby('Machine_ID')
            .last()
            .reset_index()
        )
        high_risk_machines = latest_status[latest_status['Risk_Level'] == 'High Risk']
        
        if not high_risk_machines.empty:
            fig_bar = px.bar(
                high_risk_machines,
                x='Machine_ID',
                y='Anomaly_Score',
                color='Operation_Mode',
                title="Current Anomaly Score of High-Risk Machines",
                labels={'Machine_ID': 'Machine ID', 'Anomaly_Score': 'Anomaly Score'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.success("🎉 No high-risk machines detected under current threshold parameters!")

# -----------------------------------------------------------------------------
# TAB 2: MACHINE ANOMALY DASHBOARD
# -----------------------------------------------------------------------------
with tab2:
    st.header("Machine Anomaly Dashboard")
    
    if not df_filtered.empty:
        single_machine = st.selectbox("Focus Machine:", options=sorted(df_filtered['Machine_ID'].unique()))
        m_df = df_filtered[df_filtered['Machine_ID'] == single_machine]
        
        st.subheader(f"Anomaly Score Trend: Machine {single_machine}")
        fig_line = px.line(
            m_df, 
            x='Datetime', 
            y='Anomaly_Score', 
            color='Operation_Mode',
            markers=True,
            title=f"Temporal Risk Pattern for Machine {single_machine}"
        )
        fig_line.add_hline(y=high_risk_threshold, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
        fig_line.add_hline(y=medium_risk_threshold, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
        st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("Sensor Deviation Visualization")
        sensor_choice = st.selectbox(
            "Select Sensor Metric:", 
            options=['Temperature_C', 'Vibration_Hz', 'Power_Consumption_kW', 'Error_Rate_%', 'Network_Latency_ms']
        )
        
        fig_sensor = px.scatter(
            m_df,
            x='Datetime',
            y=sensor_choice,
            color='Risk_Level',
            size='Anomaly_Score',
            color_discrete_map={'High Risk': 'red', 'Medium Risk': 'orange', 'Low Risk': 'green'},
            title=f"{sensor_choice} vs Time (Sized by Anomaly Score)"
        )
        st.plotly_chart(fig_sensor, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 3: MAINTENANCE ALERT PANEL
# -----------------------------------------------------------------------------
with tab3:
    st.header("Maintenance Alert Panel")
    
    high_alerts = df_filtered[df_filtered['Risk_Level'] == 'High Risk'].sort_values(by='Anomaly_Score', ascending=False)
    
    st.subheader(f"⚠️ Urgent Maintenance Priority List ({len(high_alerts)} Alerts)")
    
    if not high_alerts.empty:
        # Actionable recommendations generator
        def recommended_action(row):
            actions = []
            if row['Temperature_C'] > 75:
                actions.append("Inspect Cooling System")
            if row['Vibration_Hz'] > 3.0:
                actions.append("Check Mechanical Balance / Bearings")
            if row['Error_Rate_%'] > 10:
                actions.append("Calibrate Operational Alignment")
            if row['Network_Latency_ms'] > 25:
                actions.append("Verify 6G Network Module")
            return ", ".join(actions) if actions else "General Inspection Required"

        display_alerts = high_alerts[[
            'Datetime', 'Machine_ID', 'Operation_Mode', 'Anomaly_Score', 
            'Temperature_C', 'Vibration_Hz', 'Error_Rate_%'
        ]].copy()
        display_alerts['Recommended Action'] = high_alerts.apply(recommended_action, axis=1)

        st.dataframe(display_alerts, use_container_width=True)
    else:
        st.info("No active high-risk alerts at this time.")

# -----------------------------------------------------------------------------
# TAB 4: HISTORICAL RISK ANALYSIS
# -----------------------------------------------------------------------------
with tab4:
    st.header("Historical Risk & Escalation Analysis")
    
    st.subheader("Risk Escalation Timelines Across All Selected Machines")
    
    fig_heat = px.density_heatmap(
        df_filtered,
        x='Datetime',
        y='Machine_ID',
        z='Anomaly_Score',
        histfunc='avg',
        title="Machine Anomaly Score Heatmap Over Time",
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Post-Maintenance & Mode Comparison")
    fig_box = px.box(
        df_filtered,
        x='Operation_Mode',
        y='Anomaly_Score',
        color='Efficiency_Status' if 'Efficiency_Status' in df_filtered.columns else None,
        title="Anomaly Score Distribution Across Operation Modes"
    )
    st.plotly_chart(fig_box, use_container_width=True)
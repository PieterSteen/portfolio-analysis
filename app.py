import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import matplotlib.pyplot as plt
import io
from contextlib import redirect_stdout
import warnings

# Import YOUR exact analysis script
import analysis 

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. STREAMLIT HACKS (To capture notebook outputs)
# ==============================================================================
# 1A. Hijack plt.show() so Matplotlib figures render in Streamlit instead of popping up
def st_show(*args, **kwargs):
    st.pyplot(plt.gcf(), clear_figure=True)
plt.show = st_show

# 1B. Hijack IPython's display() so dataframes print nicely in the web app
def st_display(obj):
    if isinstance(obj, pd.DataFrame):
        st.dataframe(obj, use_container_width=True)
    else:
        st.write(obj)
analysis.display = st_display

# 1C. Helper function to catch print() statements and display them
def run_and_capture(func, *args, **kwargs):
    f = io.StringIO()
    with redirect_stdout(f):
        result = func(*args, **kwargs)
    
    captured_text = f.getvalue()
    if captured_text.strip():
        st.text(captured_text) # Display printed text like a terminal block
        
    return result

# ==============================================================================
# 2. APP CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(page_title="C3PM Executive Dashboard", layout="wide", page_icon="📊")

st.markdown("""
    <style>
    .metric-card {background-color: #f8f9fa; padding: 20px; border-radius: 10px; border: 1px solid #dee2e6; text-align: center; box-shadow: 2px 2px 5px rgba(0,0,0,0.05);}
    .metric-value {font-size: 28px; font-weight: bold; color: #0d6efd;}
    .metric-label {font-size: 14px; color: #6c757d; text-transform: uppercase; letter-spacing: 1px;}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 3. DATA LOADING & PREPROCESSING (Cached for Speed)
# ==============================================================================
@st.cache_data
def load_data():
    excel_path = "./input_data/032026_C3PM_Analyst_Assignment_rev0A.xlsx"
    
    # Load raw data
    df_fin = pd.read_excel(excel_path, sheet_name="financial_data")
    df_proc = pd.read_excel(excel_path, sheet_name="procurement_register")
    df_attr = pd.read_excel(excel_path, sheet_name="project_attributes")

    # Apply the Global Period 13 Fix
    df_fin['Period'] = df_fin['Period'].replace(13, 12)
    
    # Merge Financials with Attributes for filtering
    df_merged = df_fin.merge(
        df_attr[['Project Item Identifier', 'Type', 'Stage', 'Subcouncil', 'Project Manager']], 
        on='Project Item Identifier', 
        how='left'
    )
    
    # Clean up text columns
    df_merged['Type'] = df_merged['Type'].astype(str).str.title()
    df_merged['Stage'] = df_merged['Stage'].astype(str).str.title()
    
    # CPI Adjustment (to FY26 terms)
    cpi_rate = 0.05
    target_year = 2026
    df_merged['Adjusted_Value'] = df_merged['Value'] * ((1 + cpi_rate) ** (target_year - df_merged['Financial Year']))
    
    return df_merged, df_proc, df_attr, df_fin

@st.cache_data
def load_shapefile():
    try:
        return gpd.read_file("./input_data/shape_files/SL_CGIS_SUB_CNCL_2011.shp")
    except Exception as e:
        return None

try:
    df_main, df_proc, df_attr, df_fin_raw = load_data()
    gdf_subcouncils = load_shapefile()
except FileNotFoundError:
    st.error("⚠️ Dataset not found. Please ensure the Excel file is located at './input_data/032026_C3PM_Analyst_Assignment_rev0A.xlsx'.")
    st.stop()

# ==============================================================================
# 4. SIDEBAR FILTERS (Interactive Drill-down)
# ==============================================================================

try:
    st.sidebar.image("logo.png", use_container_width=True) 
except Exception as e:
    pass # Fails silently if the logo is accidentally moved or renamed

st.sidebar.title("⚙️ Portfolio Filters")
st.sidebar.markdown("Use the options below to drill down into specific portfolio segments.")



# Generate filter options safely
types_list = sorted([t for t in df_main['Type'].unique() if str(t) != 'Nan'])
stages_list = sorted([s for s in df_main['Stage'].unique() if str(s) != 'Nan'])
subcouncils_list = sorted([s for s in df_main['Subcouncil'].unique() if pd.notna(s)])

selected_types = st.sidebar.multiselect("Project Type", options=types_list, default=types_list)
selected_stages = st.sidebar.multiselect("Lifecycle Stage", options=stages_list, default=stages_list)
selected_subcouncils = st.sidebar.multiselect("Subcouncil", options=subcouncils_list, default=subcouncils_list)

# Apply Filters
mask = (
    df_main['Type'].isin(selected_types) & 
    df_main['Stage'].isin(selected_stages) & 
    df_main['Subcouncil'].isin(selected_subcouncils)
)
df_filtered = df_main[mask]

# ==============================================================================
# 5. KPI SUMMARY METRICS (FY2026 YTD)
# ==============================================================================
st.title("📊 C3PM Portfolio Executive Dashboard")
st.markdown("Monitor historical trends and FY2026 YTD spending performance.")

# Calculate KPIs for FY26 specifically (Periods 1 to 3)
fy26_data = df_filtered[df_filtered['Financial Year'] == 2026]
fy26_budget = fy26_data[fy26_data['Financial View'] == 'Original Approved Budget']['Value'].sum()
fy26_actual = fy26_data[(fy26_data['Financial View'] == 'Actual') & (fy26_data['Period'] <= 3)]['Value'].sum()
fy26_variance = fy26_budget - fy26_actual
spend_ratio = (fy26_actual / fy26_budget * 100) if fy26_budget > 0 else 0

col1, col2, col3, col4 = st.columns(4)

with col1: st.markdown(f"<div class='metric-card'><div class='metric-label'>FY26 Approved Budget</div><div class='metric-value'>R {fy26_budget/1e9:.2f} Bn</div></div>", unsafe_allow_html=True)
with col2: st.markdown(f"<div class='metric-card'><div class='metric-label'>FY26 YTD Actual Spend</div><div class='metric-value' style='color: #198754;'>R {fy26_actual/1e9:.2f} Bn</div></div>", unsafe_allow_html=True)
with col3: st.markdown(f"<div class='metric-card'><div class='metric-label'>YTD Underspend (Variance)</div><div class='metric-value' style='color: #dc3545;'>R {fy26_variance/1e9:.2f} Bn</div></div>", unsafe_allow_html=True)
with col4: st.markdown(f"<div class='metric-card'><div class='metric-label'>YTD Spend Ratio</div><div class='metric-value' style='color: {'#198754' if spend_ratio >= 25 else '#dc3545'};'>{spend_ratio:.1f}%</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Generate a cleanly filtered raw financial dataframe to pass into analysis functions
valid_project_ids = df_filtered['Project Item Identifier'].unique()
df_fin_filtered = df_fin_raw[df_fin_raw['Project Item Identifier'].isin(valid_project_ids)]

# Generate the pivot safely for downstream functions
df_fin_pivot = df_fin_filtered.pivot_table(
    index=['Project Item Identifier', 'Financial Year', 'Period'],
    columns='Financial View', values='Value', aggfunc='sum'
).reset_index().fillna(0)
df_fin_pivot.columns.name = None

# ==============================================================================
# 6. MAIN DASHBOARD TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📈 Portfolio Time-Series", "🧩 Performance by Category", "🚨 Critical Watchlist", "🧑‍💼 PM Health & Drill-Down"])

# ------------------------------------------------------------------------------
# TAB 1: TIME SERIES VIEW & NOTEBOOK ANALYTICS
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Historical & Current Spending Trends (Real, CPI Adjusted)")
    
    # Plotly Chart
    ts_pivot = df_filtered.pivot_table(index=['Financial Year', 'Period'], columns='Financial View', values='Adjusted_Value', aggfunc='sum').reset_index().fillna(0)
    
    if not ts_pivot.empty:
        ts_pivot['Time_Label'] = ts_pivot['Financial Year'].astype(str) + "-P" + ts_pivot['Period'].astype(str).str.zfill(2)
        ts_pivot.loc[(ts_pivot['Financial Year'] == 2026) & (ts_pivot['Period'] >= 4), 'Actual'] = np.nan
        
        fig_ts = go.Figure()
        if 'Original Approved Budget' in ts_pivot.columns:
            fig_ts.add_trace(go.Scatter(x=ts_pivot['Time_Label'], y=ts_pivot['Original Approved Budget'], mode='lines', name='Approved Budget', line=dict(color='orange', dash='dash')))
        if 'Actual' in ts_pivot.columns:
            fig_ts.add_trace(go.Scatter(x=ts_pivot['Time_Label'], y=ts_pivot['Actual'], mode='lines+markers', name='Actual Spend', line=dict(color='blue', width=3)))
            
        current_period = ts_pivot[(ts_pivot['Financial Year'] == 2026) & (ts_pivot['Period'] == 3)]['Time_Label'].values
        if len(current_period) > 0:
            fig_ts.add_vline(x=current_period[0], line_width=2, line_dash="dash", line_color="gray")
            fig_ts.add_annotation(x=current_period[0], y=1, yref="paper", text="End of P3", showarrow=False, font=dict(color="gray"), xanchor="left", xshift=5)
            
        fig_ts.update_layout(hovermode="x unified", xaxis_title="Financial Period", yaxis_title="Value (Rands, FY26 Terms)", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("---")
    st.subheader("Advanced Portfolio Analytics (Notebook Integration)")
    
    if not df_fin_filtered.empty:
        with st.spinner("Running Financial Trends..."):
            _, portfolio_ts = run_and_capture(analysis.plot_portfolio_financial_trends, df_fin_filtered)
            
        train_data = portfolio_ts[portfolio_ts['Period'] <= 12]
        if len(train_data) >= 24:
            with st.spinner("Running ML Forecast..."):
                ts_data, final_forecast, forecast_dates, best_model_name = run_and_capture(analysis.run_portfolio_forecast, portfolio_ts)
            with st.spinner("Running Variance..."):
                run_and_capture(analysis.plot_variance_forecast, ts_data, final_forecast, forecast_dates, best_model_name)
        else:
            st.warning("⚠️ Not enough historical data in current filter to run ML algorithms.")
            
        if gdf_subcouncils is not None:
            with st.spinner("Mapping Spatial Risk..."):
                run_and_capture(analysis.plot_spatial_risk, df_fin_filtered, df_attr, gdf_subcouncils)
    else:
        st.info("No data available.")

# ------------------------------------------------------------------------------
# TAB 2: DRILL-DOWN CATEGORIES
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("FY2026 Year-to-Date Budget vs. Actuals Breakdown")
    colA, colB = st.columns(2)
    fy26_p3 = df_filtered[(df_filtered['Financial Year'] == 2026) & (df_filtered['Period'] <= 3)]
    
    with colA:
        if not fy26_p3.empty:
            type_summary = fy26_p3.groupby(['Type', 'Financial View'])['Value'].sum().reset_index()
            fig_type = px.bar(type_summary, x='Type', y='Value', color='Financial View', barmode='group', color_discrete_map={'Original Approved Budget': 'orange', 'Actual': 'blue'}, title="YTD Spend vs Budget by Project Type")
            fig_type.update_layout(yaxis_title="Nominal Value (Rands)", xaxis_title="")
            st.plotly_chart(fig_type, use_container_width=True)
        
    with colB:
        if not fy26_p3.empty:
            stage_summary = fy26_p3.groupby(['Stage', 'Financial View'])['Value'].sum().reset_index()
            logical_order = ['Initiation', 'Concept', 'Design', 'Execution', 'Commissioning', 'Concluded']
            fig_stage = px.bar(stage_summary, x='Stage', y='Value', color='Financial View', barmode='group', category_orders={"Stage": logical_order}, color_discrete_map={'Original Approved Budget': 'orange', 'Actual': 'blue'}, title="YTD Spend vs Budget by Lifecycle Stage")
            fig_stage.update_layout(yaxis_title="Nominal Value (Rands)", xaxis_title="")
            st.plotly_chart(fig_stage, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3: EXECUTIVE WATCHLIST
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("🚨 Top 15 At-Risk Projects (Highest FY26 YTD Underspend)")
    st.markdown("These projects have the largest gap between their approved budget and actual expenditure for Periods 1 to 3.")
    
    if not fy26_p3.empty:
        proj_pivot = fy26_p3.pivot_table(index=['Project Item Identifier', 'Type', 'Stage', 'Project Manager'], columns='Financial View', values='Value', aggfunc='sum').reset_index().fillna(0)
        
        if 'Original Approved Budget' in proj_pivot.columns and 'Actual' in proj_pivot.columns:
            proj_pivot['YTD_Underspend'] = proj_pivot['Original Approved Budget'] - proj_pivot['Actual']
            watchlist = proj_pivot[proj_pivot['YTD_Underspend'] > 0].sort_values('YTD_Underspend', ascending=False).head(15)
            watchlist = watchlist[['Project Item Identifier', 'Type', 'Stage', 'Project Manager', 'Original Approved Budget', 'Actual', 'YTD_Underspend']]
            watchlist.columns = ['Project ID', 'Type', 'Current Stage', 'Project Manager', 'YTD Budget', 'YTD Actual', 'YTD Underspend Gap']
            
            st.dataframe(watchlist.style.format({'YTD Budget': 'R {:,.2f}', 'YTD Actual': 'R {:,.2f}', 'YTD Underspend Gap': 'R {:,.2f}'}).background_gradient(subset=['YTD Underspend Gap'], cmap='Reds'), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 4: PM HEALTH & DRILL-DOWN (New!)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🏆 Project Manager Spend Health Leaderboard")
    
    if not df_fin_pivot.empty:
        with st.spinner("Generating PM Leaderboard..."):
            run_and_capture(analysis.generate_pm_health_leaderboard, df_attr, df_fin_pivot)
            
        st.markdown("---")
        st.subheader("🔍 Project Manager Drill-Down Analysis")
        st.markdown("Select a specific Project Manager to view their historical lifetime spend trajectory and health scorecard.")
        
        # Build dropdown list based on PMs available in the current filter
        available_pms = sorted([pm for pm in df_filtered['Project Manager'].unique() if pd.notna(pm)])
        
        if available_pms:
            # The selectbox allows the user to filter dynamically
            selected_pm = st.selectbox("Select Project Manager:", options=available_pms)
            
            if selected_pm:
                with st.spinner(f"Analyzing Spend Health for {selected_pm}..."):
                    # Render the specific PM's graph directly from analysis.py
                    run_and_capture(analysis.analyze_pm_spend_health_drilldown, selected_pm, df_attr, df_fin_pivot)
        else:
            st.info("No Project Managers available for the current filter selection.")
    else:
        st.info("No financial data available to generate PM metrics.")
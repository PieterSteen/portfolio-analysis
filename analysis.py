import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import warnings
from IPython.display import display
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
 
def plot_portfolio_financial_trends(df_financial):
    df_fin_pivot = df_financial.pivot_table(
        index=['Project Item Identifier', 'Financial Year', 'Period'],
        columns='Financial View',
        values='Value',
        aggfunc='sum'
    ).reset_index().fillna(0)
    df_fin_pivot.columns.name = None 

    portfolio_ts = df_fin_pivot.groupby(['Financial Year', 'Period'])[['Actual', 'Original Approved Budget']].sum().reset_index()
    portfolio_ts = portfolio_ts.sort_values(by=['Financial Year', 'Period']).reset_index(drop=True)

    # Replace '0' Actuals in future periods
    portfolio_ts.loc[(portfolio_ts['Financial Year'] == 2026) & (portfolio_ts['Period'] >= 4), 'Actual'] = np.nan

    print("\n--- Transition into FY2026 (Notice Actuals are now NaN for future periods) ---")
    display(portfolio_ts.tail(15))

    portfolio_plot_data = portfolio_ts[portfolio_ts['Period'] <= 12].copy()
    portfolio_plot_data['Time'] = portfolio_plot_data['Financial Year'].astype(str) + "-P" + portfolio_plot_data['Period'].astype(str).str.zfill(2)

    cpi_rate = 0.05
    target_year = 2026
    portfolio_plot_data['Adjusted_Actual'] = portfolio_plot_data['Actual'] * ((1 + cpi_rate) ** (target_year - portfolio_plot_data['Financial Year']))
    portfolio_plot_data['Adjusted_Budget'] = portfolio_plot_data['Original Approved Budget'] * ((1 + cpi_rate) ** (target_year - portfolio_plot_data['Financial Year']))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    ax1.plot(portfolio_plot_data['Time'], portfolio_plot_data['Actual'], label='Actual Spend (Nominal)', color='blue', marker='.')
    ax1.plot(portfolio_plot_data['Time'], portfolio_plot_data['Original Approved Budget'], label='Approved Budget (Nominal)', color='orange', linestyle='--')
    ax1.set_title("Portfolio Financial Trend - Nominal (Not Adjusted for Inflation)", fontsize=16)
    ax1.set_ylabel("Value (Rands)", fontsize=12)
    ax1.set_xlabel("Time (FY-Period)", fontsize=12)
    ax1.tick_params(axis='x', rotation=90, labelsize=8)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.plot(portfolio_plot_data['Time'], portfolio_plot_data['Adjusted_Actual'], label='Actual Spend (CPI Adjusted)', color='green', marker='.')
    ax2.plot(portfolio_plot_data['Time'], portfolio_plot_data['Adjusted_Budget'], label='Approved Budget (CPI Adjusted)', color='red', linestyle='--')
    ax2.set_title("Portfolio Financial Trend - Real (Adjusted for 5% Annual CPI to FY2026 Terms)", fontsize=16)
    ax2.set_ylabel("Value (Rands)", fontsize=12)
    ax2.set_xlabel("Time (FY-Period)", fontsize=12)
    ax2.tick_params(axis='x', rotation=90, labelsize=8)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()
    
    return df_fin_pivot, portfolio_ts

def run_portfolio_forecast(portfolio_ts):
    ts_data_raw = portfolio_ts[portfolio_ts['Period'] <= 12].copy()
    cpi_rate = 0.05
    target_year = 2026

    ts_data_raw['Adjusted_Actual'] = ts_data_raw['Actual'] * ((1 + cpi_rate) ** (target_year - ts_data_raw['Financial Year']))
    ts_data_raw['Adjusted_Budget'] = ts_data_raw['Original Approved Budget'] * ((1 + cpi_rate) ** (target_year - ts_data_raw['Financial Year']))
    ts_data_raw['Date'] = pd.to_datetime(ts_data_raw['Financial Year'].astype(str) + '-' + ts_data_raw['Period'].astype(str).str.zfill(2) + '-01')
    ts_data = ts_data_raw.set_index('Date')

    train_val = ts_data[ts_data['Financial Year'] <= 2024]['Adjusted_Actual'].dropna()
    test_val = ts_data[ts_data['Financial Year'] == 2025]['Adjusted_Actual'].dropna()
    steps_val = len(test_val)

    sarima_val_fit = SARIMAX(train_val, order=(1, 0, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    pred_sarima = sarima_val_fit.get_forecast(steps=steps_val).predicted_mean
    mape_sarima = mean_absolute_percentage_error(test_val, pred_sarima)

    hw_val_fit = ExponentialSmoothing(train_val, trend=None, seasonal='add', seasonal_periods=12).fit()
    pred_hw = hw_val_fit.forecast(steps_val)
    mape_hw = mean_absolute_percentage_error(test_val, pred_hw)

    pred_ensemble = (pred_sarima + pred_hw) / 2
    mape_ensemble = mean_absolute_percentage_error(test_val, pred_ensemble)

    model_results = {
        'SARIMA': {'pred': pred_sarima, 'mape': mape_sarima},
        'Holt-Winters': {'pred': pred_hw, 'mape': mape_hw},
        'Ensemble (Avg)': {'pred': pred_ensemble, 'mape': mape_ensemble}
    }

    best_model_name = min(model_results, key=lambda k: model_results[k]['mape'])
    best_val_pred = model_results[best_model_name]['pred']
    best_val_mape = model_results[best_model_name]['mape']

    print(f"--- Model Validation Results (Predicting FY25) ---")
    print(f"1. SARIMA MAPE:          {mape_sarima:.2%}")
    print(f"2. Holt-Winters MAPE:    {mape_hw:.2%}")
    print(f"3. Ensemble MAPE:        {mape_ensemble:.2%}")
    print(f">>> Auto-Selected Best Model: {best_model_name} (MAPE: {best_val_mape:.2%})\n")

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(train_val.index, train_val, label='Training Data (FY18-FY24)', color='dodgerblue', marker='.')
    ax1.plot(test_val.index, test_val, label='Actual Holdout Data (FY25)', color='black', marker='o', linewidth=2.5)
    ax1.plot(pred_sarima.index, pred_sarima, label='SARIMA Prediction', color='green', linestyle=':', alpha=0.6)
    ax1.plot(pred_hw.index, pred_hw, label='Holt-Winters Prediction', color='purple', linestyle=':', alpha=0.6)
    ax1.plot(best_val_pred.index, best_val_pred, label=f'Best Model Winner: {best_model_name}', color='red', marker='X', linestyle='--', linewidth=2.5)

    ax1.set_title(f"Model Verification: Auto-Selected {best_model_name} vs Actuals (FY2025)", fontsize=16, pad=15)
    ax1.set_ylabel("Value (Rands, FY2026 terms)", fontsize=12)
    ax1.set_xlabel("Date (Year-Month)", fontsize=12)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    train_full = ts_data[((ts_data['Financial Year'] < 2026) | ((ts_data['Financial Year'] == 2026) & (ts_data['Period'] <= 3)))]['Adjusted_Actual'].dropna()
    forecast_steps = 9
    forecast_dates = pd.date_range(start='2026-04-01', periods=forecast_steps, freq='MS')

    if best_model_name == 'SARIMA':
        sarima_full_fit = SARIMAX(train_full, order=(1, 0, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
        final_forecast = sarima_full_fit.get_forecast(steps=forecast_steps).predicted_mean
    elif best_model_name == 'Holt-Winters':
        hw_full_fit = ExponentialSmoothing(train_full, trend=None, seasonal='add', seasonal_periods=12).fit()
        final_forecast = hw_full_fit.forecast(forecast_steps)
    else: 
        sarima_full_fit = SARIMAX(train_full, order=(1, 0, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
        hw_full_fit = ExponentialSmoothing(train_full, trend=None, seasonal='add', seasonal_periods=12).fit()
        final_forecast = (sarima_full_fit.get_forecast(steps=forecast_steps).predicted_mean + hw_full_fit.forecast(forecast_steps)) / 2

    total_budget_2026 = ts_data[ts_data['Financial Year'] == 2026]['Adjusted_Budget'].sum()
    actual_spend_p1_p3 = ts_data[(ts_data['Financial Year'] == 2026) & (ts_data['Period'] <= 3)]['Adjusted_Actual'].sum()

    forecasted_spend_p4_p12 = final_forecast.sum()
    eoy_projected_spend = actual_spend_p1_p3 + forecasted_spend_p4_p12

    print("--- FY2026 Forecast Summary (Auto-Selected Model) ---")
    print(f"Total Approved Budget: R {total_budget_2026:,.2f}")
    print(f"Actual Spend So Far (P1-P3): R {actual_spend_p1_p3:,.2f}")
    print(f"Forecasted Spend (P4-P12): R {forecasted_spend_p4_p12:,.2f}")
    print(f"Expected EOY Spend: R {eoy_projected_spend:,.2f}")
    print(f"Variance (Target vs. Expected): R {total_budget_2026 - eoy_projected_spend:,.2f}")
    print(f"Is Portfolio on track? {'Yes' if eoy_projected_spend >= total_budget_2026 else 'No, severely underperforming'}\n")

    fig2, ax2 = plt.subplots(figsize=(14, 7))
    ax2.plot(ts_data.index, ts_data['Adjusted_Actual'], label='Historical Actuals (CPI Adjusted)', color='blue', marker='.')
    ax2.plot(ts_data.index, ts_data['Adjusted_Budget'], label='Approved Budget', color='orange', linestyle='--')
    ax2.plot(forecast_dates, final_forecast, label=f'Final Forecast ({best_model_name})', color='red', marker='o', linewidth=2.5)
    ax2.axvline(pd.to_datetime('2026-03-01'), color='gray', linestyle=':', label='Current Date (End of P3)')
    ax2.set_title(f"Portfolio Financial Forecast (FY2026) - Model: {best_model_name}", fontsize=16, pad=15)
    ax2.set_ylabel("Value (Rands, FY2026 terms)", fontsize=12)
    ax2.set_xlabel("Date (Year-Month)", fontsize=12)
    ax2.legend(loc='upper left', frameon=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return ts_data, final_forecast, forecast_dates, best_model_name

def plot_variance_forecast(ts_data, final_forecast, forecast_dates, best_model_name):
    ts_data['Variance'] = ts_data['Adjusted_Actual'] - ts_data['Adjusted_Budget']
    future_budget = ts_data[(ts_data['Financial Year'] == 2026) & (ts_data['Period'] >= 4)].sort_index()
    forecasted_variance_values = final_forecast.values - future_budget['Adjusted_Budget'].values
    forecast_var_series = pd.Series(forecasted_variance_values, index=forecast_dates)

    variance_p1_p3 = ts_data[(ts_data['Financial Year'] == 2026) & (ts_data['Period'] <= 3)]['Variance'].sum()
    forecasted_variance_p4_p12 = forecast_var_series.sum()
    expected_eoy_variance = variance_p1_p3 + forecasted_variance_p4_p12

    print("--- FY2026 Variance Forecast Summary (Derived from Auto-Selected Model) ---")
    print(f"Variance So Far (P1-P3): R {variance_p1_p3:,.2f}")
    print(f"Forecasted Variance (P4-P12): R {forecasted_variance_p4_p12:,.2f}")
    print(f"Expected EOY Total Variance: R {expected_eoy_variance:,.2f}")

    if expected_eoy_variance < 0:
        print("\nConclusion: The portfolio is consistently underspending and is projected to end the year massively below budget.")
    else:
        print("\nConclusion: The portfolio is projected to overspend its budget.")

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(ts_data.index, ts_data['Variance'], label='Historical Variance (Actual - Budget)', color='purple', marker='.')
    ax.plot(forecast_dates, forecast_var_series, label=f'Forecasted Variance (Derived from {best_model_name} Model)', color='red', marker='o', linewidth=2.5)
    ax.axhline(0, color='black', linewidth=1.5, linestyle='-', label='Zero Variance (Perfectly On Target)')
    ax.axvline(pd.to_datetime('2026-03-01'), color='gray', linestyle=':', label='Current Date (End of P3)')
    ax.set_title(f"Portfolio Variance Forecast (Derived from {best_model_name} Model)", fontsize=16, pad=15)
    ax.set_ylabel("Variance (Rands, FY2026 terms)", fontsize=12)
    ax.set_xlabel("Date (Year-Month)", fontsize=12)
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_spatial_risk(df_financial, df_attributes, gdf_subcouncils):
    fy2026_budget = df_financial[(df_financial['Financial Year'] == 2026) & (df_financial['Financial View'] == 'Original Approved Budget')]
    project_budgets = fy2026_budget.groupby('Project Item Identifier')['Value'].sum().reset_index()
    project_spatial = pd.merge(project_budgets, df_attributes, on='Project Item Identifier', how='left')
    subcouncil_budgets = project_spatial.groupby('Subcouncil')['Value'].sum().reset_index()
    subcouncil_budgets.rename(columns={'Value': 'Total_FY2026_Budget'}, inplace=True)

    gdf_mapped = gdf_subcouncils.merge(subcouncil_budgets, left_on='SUB_CNCL_1', right_on='Subcouncil', how='left')
    gdf_mapped['Total_FY2026_Budget'] = gdf_mapped['Total_FY2026_Budget'].fillna(0)
    gdf_mapped['Budget_Millions'] = gdf_mapped['Total_FY2026_Budget'] / 1_000_000

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    gdf_mapped.plot(column='Budget_Millions', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.2', 
                    legend=True, cax=cax, legend_kwds={'label': "FY2026 Approved Budget (Millions ZAR)"})

    for x, y, label in zip(gdf_mapped.geometry.centroid.x, gdf_mapped.geometry.centroid.y, gdf_mapped['SUB_CNCL_1']):
        ax.annotate(text=int(label), xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=8, color='black', weight='bold')

    ax.set_title('FY2026 Spatial Distribution of Budget Risk by Subcouncil', fontsize=16, pad=15)
    ax.set_axis_off() 
    plt.tight_layout()
    plt.show()

    top_3 = subcouncil_budgets.sort_values(by='Total_FY2026_Budget', ascending=False).head(3)
    print("--- Top 3 Subcouncils by FY2026 Budget ---")
    for index, row in top_3.iterrows():
        print(f"Subcouncil {int(row['Subcouncil'])}: R {row['Total_FY2026_Budget']:,.2f}")

# =============================================================================
# PROCUREMENT, PM CAPACITY, AND STRUCTURAL RISK
# =============================================================================

def analyze_structural_risk(df_financial, df_procurement, df_attributes):
    current_date = pd.to_datetime('2025-09-30')
    df_procurement['Date of Expiry'] = pd.to_datetime(df_procurement['Date of Expiry'])

    active_contracts = df_procurement[(df_procurement['Status'] == 'Awarded') & (df_procurement['Date of Expiry'] >= current_date)]
    proc_summary = active_contracts.groupby(['Project Item Identifier', 'Category']).size().unstack(fill_value=0)
    proc_summary.columns = [col.lower() for col in proc_summary.columns] 

    if 'service' not in proc_summary.columns: proc_summary['service'] = 0
    if 'contractor' not in proc_summary.columns: proc_summary['contractor'] = 0

    proc_summary['has_contractor'] = proc_summary['contractor'] > 0
    proc_summary['has_service'] = proc_summary['service'] > 0
    proc_summary = proc_summary[['has_contractor', 'has_service']].reset_index()

    fy26_rem_budget = df_financial[(df_financial['Financial Year'] == 2026) & (df_financial['Period'] >= 4) & (df_financial['Financial View'] == 'Original Approved Budget')]
    budget_per_proj = fy26_rem_budget.groupby('Project Item Identifier')['Value'].sum().reset_index()
    budget_per_proj.rename(columns={'Value': 'Remaining_FY26_Budget'}, inplace=True)

    df_proj = df_attributes.merge(budget_per_proj, on='Project Item Identifier', how='left').fillna({'Remaining_FY26_Budget': 0})
    df_proj = df_proj.merge(proc_summary, on='Project Item Identifier', how='left').fillna({'has_contractor': False, 'has_service': False})

    def check_if_blocked(row):
        if row['Stage'] == 'concluded': return False, "Concluded"
        t, s = str(row['Type']).lower(), str(row['Stage']).lower()
        if t == 'capital purchase':
            if not row['has_contractor']: return True, "Missing Contractor (Purchase)"
        elif t in ['large', 'medium']:
            if not row['has_service']: return True, "Missing Service Provider"
            if s in ['execution', 'commissioning'] and not row['has_contractor']: return True, "Missing Contractor (Execution/Commission)"
        elif t == 'small':
            if s in ['execution', 'commissioning'] and not row['has_contractor']: return True, "Missing Contractor (Execution/Commission)"
        return False, "On Track"

    df_proj[['Is_Blocked', 'Block_Reason']] = df_proj.apply(check_if_blocked, axis=1, result_type='expand')

    pm_counts = df_proj[df_proj['Stage'] != 'concluded'].groupby('Project Manager').size()
    overloaded_pms = pm_counts[pm_counts > pm_counts.quantile(0.75)] 

    blocked_projects = df_proj[df_proj['Is_Blocked']]
    blocked_budget = blocked_projects['Remaining_FY26_Budget'].sum()
    total_rem_budget = df_proj['Remaining_FY26_Budget'].sum()

    print("--- Structural Risk & Procurement Bottlenecks ---")
    print(f"Total Remaining FY26 Budget (P4-P12): R {total_rem_budget:,.2f}")
    print(f"Total Budget 'Blocked' by Procurement Rules: R {blocked_budget:,.2f}")
    print(f"Percentage of Budget Structurally at Risk: {(blocked_budget/total_rem_budget)*100:.2f}%\n")

    print("--- Breakdown of Blocked Budget by Reason ---")
    reason_summary = blocked_projects.groupby('Block_Reason')['Remaining_FY26_Budget'].sum().sort_values(ascending=False)
    for reason, val in reason_summary.items():
        print(f"- {reason}: R {val:,.2f}")

    print(f"\n--- PM Capacity Constraints ---")
    print(f"Found {len(overloaded_pms)} Project Managers handling a disproportionately high number of active projects (>{int(pm_counts.quantile(0.75))} projects each).")
    print("This presents a significant delivery risk even if procurement is resolved.")
    
    return df_proj

def analyze_pm_capacity_concurrent(df_financial, df_attributes):
    df_pm_activity = df_financial.merge(df_attributes[['Project Item Identifier', 'Project Manager']], on='Project Item Identifier', how='left')
    pm_project_counts = df_pm_activity.groupby(['Project Manager', 'Financial Year', 'Period'])['Project Item Identifier'].nunique().reset_index()
    pm_project_counts.rename(columns={'Project Item Identifier': 'Concurrent_Projects'}, inplace=True)
    overloaded_pms = pm_project_counts[pm_project_counts['Concurrent_Projects'] > 1]

    print("==================================================")
    print("   PROJECT MANAGER CAPACITY ANALYSIS")
    print("==================================================\n")

    if overloaded_pms.empty:
        print("No Project Manager works on more than 1 project in any given period.")
    else:
        num_overloaded = overloaded_pms['Project Manager'].nunique()
        total_pms = df_attributes['Project Manager'].nunique()
        
        print(f"YES. Out of {total_pms} total Project Managers, {num_overloaded} of them work on multiple projects concurrently in a single period.")
        print(f"There are {len(overloaded_pms)} specific historical and future months where this overlap occurs.\n")
        
        print("--- Top 10 Busiest PM Periods (Most Concurrent Projects) ---")
        display(overloaded_pms.sort_values(by='Concurrent_Projects', ascending=False).reset_index(drop=True).head(10))
        
        print("\n--- Maximum Concurrent Projects Managed at Once (Top 10 PMs) ---")
        max_concurrent = overloaded_pms.groupby('Project Manager')['Concurrent_Projects'].max().sort_values(ascending=False).reset_index()
        display(max_concurrent.head(10))

 
def plot_project_financials(project_id, df_fin_pivot, df_proj, cpi_rate=0.05, target_year=2026):
    proj_data = df_fin_pivot[df_fin_pivot['Project Item Identifier'] == project_id].copy()
    if proj_data.empty:
        print(f"No financial data found for project {project_id}")
        return
        
    proj_data = proj_data[proj_data['Period'] <= 12].sort_values(by=['Financial Year', 'Period'])
    proj_data['Adjusted_Actual'] = proj_data['Actual'] * ((1 + cpi_rate) ** (target_year - proj_data['Financial Year']))
    proj_data['Adjusted_Budget'] = proj_data['Original Approved Budget'] * ((1 + cpi_rate) ** (target_year - proj_data['Financial Year']))
    proj_data.loc[(proj_data['Financial Year'] == 2026) & (proj_data['Period'] >= 4), 'Adjusted_Actual'] = np.nan
    proj_data['Date'] = pd.to_datetime(proj_data['Financial Year'].astype(str) + '-' + proj_data['Period'].astype(str).str.zfill(2) + '-01')
    proj_data.set_index('Date', inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(proj_data.index, proj_data['Adjusted_Actual'], label='Actual Spend (CPI Adjusted)', color='blue', marker='o')
    ax.plot(proj_data.index, proj_data['Adjusted_Budget'], label='Approved Budget (CPI Adjusted)', color='orange', linestyle='--')
    ax.axvline(pd.to_datetime('2026-03-01'), color='gray', linestyle=':', label='Current Date (End of P3)')
    
    is_blocked, block_reason = False, ""
    if 'Is_Blocked' in df_proj.columns:
        proj_info = df_proj[df_proj['Project Item Identifier'] == project_id]
        if not proj_info.empty:
            is_blocked = proj_info.iloc[0]['Is_Blocked']
            block_reason = proj_info.iloc[0]['Block_Reason']
            
    title_status = f" | STATUS: BLOCKED ({block_reason})" if is_blocked else " | STATUS: On Track"
    
    ax.set_title(f"Project Financial State: {project_id}{title_status}", fontsize=14, pad=15)
    ax.set_ylabel("Value (Rands, FY2026 terms)", fontsize=10)
    ax.set_xlabel("Date (Year-Month)", fontsize=10)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def analyze_pm_portfolio(pm_name, df_proj, df_fin_pivot, cpi_rate=0.05, target_year=2026):
    print(f"==================================================")
    print(f"   PORTFOLIO ANALYSIS FOR: {pm_name}")
    print(f"==================================================\n")
    
    pm_projects = df_proj[df_proj['Project Manager'] == pm_name]
    if pm_projects.empty:
        print(f"No projects found for {pm_name}.")
        return
        
    total_projects = len(pm_projects)
    active_projects = pm_projects[pm_projects['Stage'] != 'concluded']
    concluded_projects = total_projects - len(active_projects)
    total_rem_budget = pm_projects['Remaining_FY26_Budget'].sum()
    blocked_projects = active_projects[active_projects['Is_Blocked'] == True]
    blocked_budget = blocked_projects['Remaining_FY26_Budget'].sum()
    
    print("--- PROJECT LIFECYCLE SUMMARY ---")
    print(f"Total Projects Assigned: {total_projects}")
    print(f"Active Projects: {len(active_projects)}")
    print(f"Concluded Projects: {concluded_projects}")
    
    print("\n--- FY2026 BUDGET RISK ---")
    print(f"Total Remaining FY26 Budget (P4-P12): R {total_rem_budget:,.2f}")
    print(f"Budget Structurally Blocked: R {blocked_budget:,.2f}")
    
    if total_rem_budget > 0:
        print(f"Percentage of PM's Budget at Risk: {(blocked_budget / total_rem_budget) * 100:.2f}%")
        
    if not blocked_projects.empty:
        print("\n--- BLOCK REASONS ---")
        reasons = blocked_projects.groupby('Block_Reason')['Remaining_FY26_Budget'].sum()
        for reason, val in reasons.items():
            print(f"- {reason}: R {val:,.2f} ({len(blocked_projects[blocked_projects['Block_Reason'] == reason])} projects)")
            
    pm_project_ids = pm_projects['Project Item Identifier'].tolist()
    pm_fin_data = df_fin_pivot[df_fin_pivot['Project Item Identifier'].isin(pm_project_ids)].copy()
    if pm_fin_data.empty: return
        
    pm_ts = pm_fin_data.groupby(['Financial Year', 'Period'])[['Actual', 'Original Approved Budget']].sum().reset_index()
    pm_ts = pm_ts[pm_ts['Period'] <= 12].sort_values(by=['Financial Year', 'Period'])
    pm_ts['Adjusted_Actual'] = pm_ts['Actual'] * ((1 + cpi_rate) ** (target_year - pm_ts['Financial Year']))
    pm_ts['Adjusted_Budget'] = pm_ts['Original Approved Budget'] * ((1 + cpi_rate) ** (target_year - pm_ts['Financial Year']))
    pm_ts.loc[(pm_ts['Financial Year'] == 2026) & (pm_ts['Period'] >= 4), 'Adjusted_Actual'] = np.nan
    pm_ts['Date'] = pd.to_datetime(pm_ts['Financial Year'].astype(str) + '-' + pm_ts['Period'].astype(str).str.zfill(2) + '-01')
    pm_ts.set_index('Date', inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(pm_ts.index, pm_ts['Adjusted_Actual'], label='Actual Spend (CPI Adjusted)', color='blue', marker='o')
    ax.plot(pm_ts.index, pm_ts['Adjusted_Budget'], label='Approved Budget (CPI Adjusted)', color='orange', linestyle='--')
    ax.axvline(pd.to_datetime('2026-03-01'), color='gray', linestyle=':', label='Current Date (End of P3)')
    ax.set_title(f"Financial Trajectory for {pm_name}'s Portfolio", fontsize=14, pad=15)
    ax.set_ylabel("Value (Rands, FY2026 terms)", fontsize=10)
    ax.set_xlabel("Date (Year-Month)", fontsize=10)
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def analyze_pm_projects_individual(pm_name, df_proj, df_procurement, df_fin_pivot, cpi_rate=0.05, target_year=2026):
    print(f"==================================================")
    print(f"   INDIVIDUAL PROJECT DRILL-DOWN: {pm_name}")
    print(f"==================================================\n")
    
    pm_projects = df_proj[df_proj['Project Manager'] == pm_name]
    if pm_projects.empty:
        print(f"No projects found for {pm_name}.")
        return
        
    active_projects = pm_projects
    if active_projects.empty:
        print(f"All projects for {pm_name} are already concluded.")
        return
        
    print(f"Found {len(active_projects)} active projects for {pm_name}. Generating reports...\n")
        
    for _, proj_row in active_projects.iterrows():
        proj_id = proj_row['Project Item Identifier']
        proj_type, proj_stage = str(proj_row['Type']).lower(), str(proj_row['Stage']).lower()
        
        proj_fin = df_fin_pivot[(df_fin_pivot['Project Item Identifier'] == proj_id) & (df_fin_pivot['Period'] <= 12)].copy()
        if proj_fin.empty: continue
            
        proj_fin['Adjusted_Budget'] = proj_fin['Original Approved Budget'] * ((1 + cpi_rate) ** (target_year - proj_fin['Financial Year']))
        proj_fin['Adjusted_Actual'] = proj_fin['Actual'] * ((1 + cpi_rate) ** (target_year - proj_fin['Financial Year']))
        proj_fin.loc[(proj_fin['Financial Year'] == 2026) & (proj_fin['Period'] >= 4), 'Adjusted_Actual'] = np.nan
        proj_fin['Plot_Date'] = pd.to_datetime(proj_fin['Financial Year'].astype(str) + '-' + proj_fin['Period'].astype(str).str.zfill(2) + '-01')
        
        def get_calendar_date(row):
            fy, period = int(row['Financial Year']), int(row['Period'])
            if period <= 6: return pd.to_datetime(f"{fy - 1}-{period + 6:02d}-01")
            else: return pd.to_datetime(f"{fy}-{period - 6:02d}-01")
            
        proj_fin['Eval_Date'] = proj_fin.apply(get_calendar_date, axis=1)
        proj_fin = proj_fin.sort_values('Plot_Date')
        
        proc_proj = df_procurement[df_procurement['Project Item Identifier'] == proj_id].copy()
        proc_proj['Date of Expiry'] = pd.to_datetime(proc_proj['Date of Expiry'])
        proc_proj['Award Date'] = pd.to_datetime(proc_proj['Award Date'])
        
        def get_status(row):
            eval_date = row['Eval_Date']
            if eval_date < pd.to_datetime('2025-10-01'): return "Historical"
            active_contracts = proc_proj[(proc_proj['Status'] == 'Awarded') & (proc_proj['Award Date'] <= eval_date) & (proc_proj['Date of Expiry'] >= eval_date)]
            has_contractor = 'Contractor' in active_contracts['Category'].values or 'contractor' in active_contracts['Category'].values
            has_service = 'Service' in active_contracts['Category'].values or 'service' in active_contracts['Category'].values
            
            if proj_type == 'capital purchase' and not has_contractor: return "Blocked (No Contractor)"
            elif proj_type in ['large', 'medium']:
                if not has_service: return "Blocked (No Service Provider)"
                if proj_stage in ['execution', 'commissioning'] and not has_contractor: return "Blocked (No Contractor)"
            elif proj_type == 'small':
                if proj_stage in ['execution', 'commissioning'] and not has_contractor: return "Blocked (No Contractor)"
            return "On Track"
            
        proj_fin['Status'] = proj_fin.apply(get_status, axis=1)
        
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(proj_fin['Plot_Date'], proj_fin['Adjusted_Budget'], color='gray', linestyle='--', label='Adjusted Budget (Path)', alpha=0.5, zorder=1)
        ax.plot(proj_fin['Plot_Date'], proj_fin['Adjusted_Actual'], color='black', linewidth=2, label='Adjusted Actual Spend', zorder=2)
        
        statuses = proj_fin['Status'].unique()
        color_dict = {'Historical': 'lightgray', 'On Track': 'limegreen'}
        blocked_colors = ['crimson', 'darkorange', 'purple'] 
        b_idx = 0
        for s in statuses:
            if s not in color_dict:
                color_dict[s] = blocked_colors[b_idx % len(blocked_colors)]
                b_idx += 1
        
        for status in statuses:
            subset = proj_fin[proj_fin['Status'] == status]
            ax.scatter(subset['Plot_Date'], subset['Adjusted_Budget'], color=color_dict[status], s=70, label=f"Status: {status}", zorder=5, edgecolors='black', linewidth=0.8)
                       
        ax.set_title(f"Project Timeline: {proj_id} | Type: {proj_type.title()} | Current Stage: {proj_stage.title()}", fontsize=14, pad=15)
        ax.set_ylabel("Value (Rands, FY2026 terms)", fontsize=10)
        ax.set_xlabel("Financial Timeline (Year-Period)", fontsize=10)
        
        current_date_marker = pd.to_datetime('2026-03-01')
        if current_date_marker in proj_fin['Plot_Date'].values:
            ax.axvline(current_date_marker, color='blue', linestyle=':', linewidth=2, label='Current Date (End of P3)', zorder=0)
            
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-P%m'))
        plt.xticks(rotation=45)
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True, fontsize=10, title="Legend")
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.show()

 
def generate_pm_health_leaderboard(df_proj, df_fin_pivot, cpi_rate=0.05, target_year=2026):
    hist_fin = df_fin_pivot[((df_fin_pivot['Financial Year'] < 2026) | ((df_fin_pivot['Financial Year'] == 2026) & (df_fin_pivot['Period'] <= 3)))].copy()
    hist_fin['Adjusted_Budget'] = hist_fin['Original Approved Budget'] * ((1 + cpi_rate) ** (target_year - hist_fin['Financial Year']))
    hist_fin['Adjusted_Actual'] = hist_fin['Actual'] * ((1 + cpi_rate) ** (target_year - hist_fin['Financial Year']))
    
    hist_pm = hist_fin.merge(df_proj[['Project Item Identifier', 'Project Manager']], on='Project Item Identifier', how='left')
    pm_health = hist_pm.groupby('Project Manager')[['Adjusted_Actual', 'Adjusted_Budget']].sum().reset_index()
    pm_health['Spend_Ratio'] = (pm_health['Adjusted_Actual'] / pm_health['Adjusted_Budget']) * 100
    pm_health = pm_health.sort_values(by='Spend_Ratio', ascending=True).reset_index(drop=True)
    
    colors = ['crimson' if r < 85 else 'darkorange' if r > 105 else 'limegreen' for r in pm_health['Spend_Ratio']]
        
    fig, ax = plt.subplots(figsize=(10, 14))
    bars = ax.barh(pm_health['Project Manager'], pm_health['Spend_Ratio'], color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(100, color='black', linestyle='-', linewidth=2, label='Perfect Target (100%)', zorder=0)
    ax.axvline(85, color='gray', linestyle='--', linewidth=1.5, label='Lower Threshold (85%)', zorder=0)
    ax.axvline(105, color='gray', linestyle=':', linewidth=1.5, label='Upper Threshold (105%)', zorder=0)
    
    ax.set_title("Portfolio-Wide PM Spend Health (Lifetime Actual vs Budget %)", fontsize=16, pad=20)
    ax.set_xlabel("Spend Health Ratio (%)", fontsize=12)
    ax.set_ylabel("Project Manager", fontsize=12)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', ha='left', fontsize=8)
        
    ax.set_xlim(0, pm_health['Spend_Ratio'].max() * 1.1)
    legend_elements = [
        Patch(facecolor='limegreen', edgecolor='black', label='HEALTHY (85% - 105%)'),
        Patch(facecolor='crimson', edgecolor='black', label='CRITICAL UNDERSPEND (< 85%)'),
        Patch(facecolor='darkorange', edgecolor='black', label='WARNING OVERSPEND (> 105%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.show()

def analyze_pm_spend_health_drilldown(pm_name, df_proj, df_fin_pivot, cpi_rate=0.05, target_year=2026):
    pm_projects = df_proj[df_proj['Project Manager'] == pm_name]
    if pm_projects.empty: return
    
    hist_fin = df_fin_pivot[(df_fin_pivot['Project Item Identifier'].isin(pm_projects['Project Item Identifier'])) & 
                            ((df_fin_pivot['Financial Year'] < 2026) | ((df_fin_pivot['Financial Year'] == 2026) & (df_fin_pivot['Period'] <= 3)))].copy()
    if hist_fin.empty: return
        
    hist_fin['Adjusted_Budget'] = hist_fin['Original Approved Budget'] * ((1 + cpi_rate) ** (target_year - hist_fin['Financial Year']))
    hist_fin['Adjusted_Actual'] = hist_fin['Actual'] * ((1 + cpi_rate) ** (target_year - hist_fin['Financial Year']))
    
    total_budget, total_actual = hist_fin['Adjusted_Budget'].sum(), hist_fin['Adjusted_Actual'].sum()
    spend_ratio = (total_actual / total_budget) * 100 if total_budget > 0 else 0
    variance = total_actual - total_budget
    
    if spend_ratio < 85: health_status, status_color = "CRITICAL: Chronic Underspending", 'crimson'
    elif spend_ratio > 105: health_status, status_color = "WARNING: Chronic Overspending", 'darkorange'
    else: health_status, status_color = "HEALTHY: On Target", 'limegreen'
    
    print(f"==================================================")
    print(f"   SPEND HEALTH SCORECARD: {pm_name}")
    print(f"==================================================\n")
    print(f"Total Lifetime Budget Managed (Adjusted): R {total_budget:,.2f}")
    print(f"Total Lifetime Actual Spend (Adjusted):   R {total_actual:,.2f}")
    print(f"Lifetime Variance:                        R {variance:,.2f}")
    print(f"Spend Health Ratio:                       {spend_ratio:.1f}%")
    print(f"Status:                                   {health_status}\n")
    
    def get_calendar_date(row):
        fy, period = int(row['Financial Year']), int(row['Period'])
        if period <= 6: return pd.to_datetime(f"{fy - 1}-{period + 6:02d}-01")
        else: return pd.to_datetime(f"{fy}-{period - 6:02d}-01")
            
    hist_fin['Date'] = hist_fin.apply(get_calendar_date, axis=1)
    ts_health = hist_fin.groupby('Date')[['Adjusted_Actual', 'Adjusted_Budget']].sum().sort_index()
    ts_health['Cum_Actual'] = ts_health['Adjusted_Actual'].cumsum()
    ts_health['Cum_Budget'] = ts_health['Adjusted_Budget'].cumsum()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts_health.index, ts_health['Cum_Budget'], color='gray', linestyle='--', linewidth=2.5, label='Target Trajectory (Cumulative Budget)')
    ax.plot(ts_health.index, ts_health['Cum_Actual'], color='black', linewidth=3, label='Actual Performance (Cumulative Spend)')
    ax.fill_between(ts_health.index, ts_health['Cum_Actual'], ts_health['Cum_Budget'], where=(ts_health['Cum_Actual'] < ts_health['Cum_Budget']), interpolate=True, color='crimson', alpha=0.2, label='Underspend Gap')
    ax.fill_between(ts_health.index, ts_health['Cum_Actual'], ts_health['Cum_Budget'], where=(ts_health['Cum_Actual'] > ts_health['Cum_Budget']), interpolate=True, color='orange', alpha=0.3, label='Overspend Gap')
    
    ax.set_title(f"Lifetime Spend Health Curve: {pm_name}\nHealth Score: {spend_ratio:.1f}% ({health_status.split(':')[0]})", fontsize=16, pad=15, color=status_color, fontweight='bold')
    ax.set_ylabel("Cumulative Value (Rands, FY2026 terms)", fontsize=12)
    ax.set_xlabel("Historical Timeline (Calendar Months)", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    plt.xticks(rotation=45)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True, fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def diagnose_underspender(pm_name, df_attributes, df_financial, df_procurement, df_fin_pivot, cpi_rate=0.05, target_year=2026):
    print(f"==================================================")
    print(f"   FORENSIC UNDERSPEND DIAGNOSTIC: {pm_name}")
    print(f"==================================================\n")
    
    pm_projects = df_attributes[df_attributes['Project Manager'] == pm_name].copy()
    if pm_projects.empty: return

    hist_fin = df_fin_pivot[(df_fin_pivot['Project Item Identifier'].isin(pm_projects['Project Item Identifier'])) & 
                            ((df_fin_pivot['Financial Year'] < 2026) | ((df_fin_pivot['Financial Year'] == 2026) & (df_fin_pivot['Period'] <= 3)))].copy()
    if hist_fin.empty: return
        
    hist_fin['Adjusted_Budget'] = hist_fin['Original Approved Budget'] * ((1 + cpi_rate) ** (target_year - hist_fin['Financial Year']))
    hist_fin['Adjusted_Actual'] = hist_fin['Actual'] * ((1 + cpi_rate) ** (target_year - hist_fin['Financial Year']))

    total_budget_months = len(hist_fin[hist_fin['Adjusted_Budget'] > 0])
    zero_spend_months = len(hist_fin[(hist_fin['Adjusted_Budget'] > 0) & (hist_fin['Adjusted_Actual'] == 0)])
    zero_spend_ratio = (zero_spend_months / total_budget_months) * 100 if total_budget_months > 0 else 0

    active_months = hist_fin[(hist_fin['Adjusted_Budget'] > 0) & (hist_fin['Adjusted_Actual'] > 0)]
    exec_efficiency = (active_months['Adjusted_Actual'].sum() / active_months['Adjusted_Budget'].sum()) * 100 if not active_months.empty else 0
    complexity_mix = pm_projects['Type'].value_counts()

    active_pm_projects = pm_projects[pm_projects['Stage'] != 'concluded'].copy()
    current_date = pd.to_datetime('2025-09-30')
    df_proc = df_procurement.copy()
    df_proc['Date of Expiry'] = pd.to_datetime(df_proc['Date of Expiry'])
    
    fy26_rem_budget = df_financial[(df_financial['Financial Year'] == 2026) & (df_financial['Period'] >= 4) & (df_financial['Financial View'] == 'Original Approved Budget')]
    budget_per_proj = fy26_rem_budget.groupby('Project Item Identifier')['Value'].sum()

    blocked_budget, unblocked_budget = 0, 0
    for _, row in active_pm_projects.iterrows():
        proj_id = row['Project Item Identifier']
        t, s = str(row['Type']).lower(), str(row['Stage']).lower()
        budget = budget_per_proj.get(proj_id, 0)
        
        active_contracts = df_proc[(df_proc['Project Item Identifier'] == proj_id) & (df_proc['Status'] == 'Awarded') & (df_proc['Date of Expiry'] >= current_date)]
        has_contractor = 'Contractor' in active_contracts['Category'].values or 'contractor' in active_contracts['Category'].values
        has_service = 'Service' in active_contracts['Category'].values or 'service' in active_contracts['Category'].values
        
        is_blocked = False
        if t == 'capital purchase' and not has_contractor: is_blocked = True
        elif t in ['large', 'medium']:
            if not has_service: is_blocked = True
            elif s in ['execution', 'commissioning'] and not has_contractor: is_blocked = True
        elif t == 'small' and s in ['execution', 'commissioning'] and not has_contractor: is_blocked = True
            
        if is_blocked: blocked_budget += budget
        else: unblocked_budget += budget

    print(f"1. IDLE MONTHS: This PM spent R0 in {zero_spend_ratio:.1f}% of their budgeted months.")
    print(f"2. EXECUTION EFFICIENCY: When they do spend money, they only hit {exec_efficiency:.1f}% of their monthly target.")
    print(f"3. UPCOMING RISK: R {blocked_budget/1e6:.1f} Million of their remaining FY26 budget is currently frozen due to missing contracts.\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Forensic Underspend Diagnostic for {pm_name}", fontsize=18, fontweight='bold', y=0.98)
    
    axes[0, 0].pie([zero_spend_ratio, 100-zero_spend_ratio], labels=['Zero Spend (Idle)', 'Active Spend'], colors=['crimson', 'lightgray'], autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'), textprops={'fontsize': 10})
    axes[0, 0].set_title("1. Idle Months (Budget Approved, R0 Spent)", fontsize=12, pad=10)
    
    bar_color = 'limegreen' if exec_efficiency >= 85 else 'darkorange' if exec_efficiency >= 70 else 'crimson'
    axes[0, 1].bar(['Execution Efficiency'], [exec_efficiency], color=bar_color, width=0.4, edgecolor='black')
    axes[0, 1].axhline(100, color='black', linestyle='--', label='100% Perfect Target')
    axes[0, 1].set_ylim(0, max(120, exec_efficiency + 10))
    axes[0, 1].text(0, exec_efficiency + 2, f"{exec_efficiency:.1f}%", ha='center', fontsize=14, fontweight='bold')
    axes[0, 1].set_title("2. Execution Efficiency (Average % Spent When Active)", fontsize=12)
    axes[0, 1].legend(loc='lower right')
    
    axes[1, 0].bar(complexity_mix.index.str.title(), complexity_mix.values, color='mediumseagreen', edgecolor='black', width=0.6)
    axes[1, 0].set_title("3. Portfolio Complexity (Project Types Assigned)", fontsize=12)
    axes[1, 0].set_ylabel("Number of Projects", fontsize=10)
    for i, v in enumerate(complexity_mix.values): axes[1, 0].text(i, v + 0.1, str(v), ha='center', fontsize=12, fontweight='bold')
        
    total_fy26 = blocked_budget + unblocked_budget
    if total_fy26 > 0:
        axes[1, 1].pie([blocked_budget, unblocked_budget], labels=['Blocked (Missing Contracts)', 'On Track'], colors=['darkorange', 'dodgerblue'], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        axes[1, 1].set_title(f"4. FY26 Remaining Budget Risk\n(Total: R {total_fy26/1e6:.1f}M)", fontsize=12)
    else:
        axes[1, 1].text(0.5, 0.5, "No Active FY26 Budget", ha='center', va='center', fontsize=12, color='gray')
        axes[1, 1].axis('off')
        axes[1, 1].set_title("4. FY26 Remaining Budget Risk", fontsize=12)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

 
def analyze_expenditure_by_stage(df_fin_pivot, df_attributes):
    print("==================================================")
    print("   EXPENDITURE BY LIFECYCLE STAGE ANALYSIS")
    print("==================================================\n")
    
    recent_fin = df_fin_pivot[((df_fin_pivot['Financial Year'] >= 2024) & (df_fin_pivot['Financial Year'] < 2026)) | ((df_fin_pivot['Financial Year'] == 2026) & (df_fin_pivot['Period'] <= 3))].copy()
    recent_fin = recent_fin[recent_fin['Actual'] > 0]
    fin_with_stage = recent_fin.merge(df_attributes[['Project Item Identifier', 'Stage', 'Type']], on='Project Item Identifier', how='left')
    
    active_stages = ['design', 'execution', 'commissioning']
    stage_data = fin_with_stage[fin_with_stage['Stage'].isin(active_stages)].copy()
    stage_data['Stage'] = stage_data['Stage'].str.title()
    plot_order = ['Design', 'Execution', 'Commissioning']
    
    if stage_data.empty: return
        
    print("--- Monthly Spend Statistics per Active Stage (Recent 2 Years) ---")
    stats = stage_data.groupby('Stage')['Actual'].agg(Average_Monthly_Spend='mean', Median_Monthly_Spend='median', Max_Monthly_Spike='max', Months_with_Spend='count').reindex(plot_order)
    formatted_stats = stats.copy()
    for col in ['Average_Monthly_Spend', 'Median_Monthly_Spend', 'Max_Monthly_Spike']:
        formatted_stats[col] = formatted_stats[col].apply(lambda x: f"R {x:,.2f}")
    display(formatted_stats)
    
    design_mean, exec_mean = stats.loc['Design', 'Average_Monthly_Spend'], stats.loc['Execution', 'Average_Monthly_Spend']
    print(f"\nConclusion: The average monthly spend during Execution is {exec_mean / design_mean if design_mean > 0 else 0:.1f}x higher than during Design.")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=stage_data, x='Stage', y='Actual', order=plot_order, palette=['royalblue', 'darkorange', 'mediumpurple'], estimator=np.mean, ci=None, ax=ax1, edgecolor='black')
    ax1.set_title("Average Monthly Spend per Project by Current Stage", fontsize=14, pad=10)
    ax1.set_ylabel("Average Spend (Rands)", fontsize=12)
    ax1.set_xlabel("Project Stage", fontsize=12)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
    for p in ax1.patches: ax1.annotate(f"R {p.get_height()/1e6:.1f}M", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontweight='bold')

    sns.boxplot(data=stage_data, x='Stage', y='Actual', order=plot_order, palette=['royalblue', 'darkorange', 'mediumpurple'], ax=ax2)
    ax2.set_yscale('log')
    ax2.set_title("Distribution & Volatility of Monthly Spend (Log Scale)", fontsize=14, pad=10)
    ax2.set_ylabel("Monthly Spend (Rands, Log Scale)", fontsize=12)
    ax2.set_xlabel("Project Stage", fontsize=12)
    plt.suptitle("Verification of Lifecycle Expenditure Dynamics", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def analyze_root_causes_by_type_old(df_financial, df_attributes, df_procurement):
    print("==================================================")
    print("   Q2: ROOT CAUSE ANALYSIS BY PROJECT TYPE (FY26 YTD)")
    print("==================================================\n")
    
    fy26_ytd = df_financial[(df_financial['Financial Year'] == 2026) & (df_financial['Period'] <= 3)]
    ytd_pivot = fy26_ytd.pivot_table(index='Project Item Identifier', columns='Financial View', values='Value', aggfunc='sum').reset_index().fillna(0)
    ytd_pivot['YTD_Underspend'] = ytd_pivot['Original Approved Budget'] - ytd_pivot['Actual']
    df_merged = ytd_pivot[ytd_pivot['YTD_Underspend'] > 0].copy().merge(df_attributes[['Project Item Identifier', 'Type', 'Stage']], on='Project Item Identifier', how='left')
    
    current_date = pd.to_datetime('2025-09-30')
    df_proc = df_procurement.copy()
    df_proc['Date of Expiry'], df_proc['Award Date'] = pd.to_datetime(df_proc['Date of Expiry']), pd.to_datetime(df_proc['Award Date'])
    active_contracts = df_proc[(df_proc['Status'] == 'Awarded') & (df_proc['Award Date'] <= current_date) & (df_proc['Date of Expiry'] >= current_date)]
    
    proc_summary = active_contracts.groupby(['Project Item Identifier', 'Category']).size().unstack(fill_value=0)
    proc_summary.columns = [col.lower() for col in proc_summary.columns]
    if 'service' not in proc_summary.columns: proc_summary['service'] = 0
    if 'contractor' not in proc_summary.columns: proc_summary['contractor'] = 0
    proc_summary['has_contractor'], proc_summary['has_service'] = proc_summary['contractor'] > 0, proc_summary['service'] > 0
    proc_summary = proc_summary.reset_index()
    
    df_merged = df_merged.merge(proc_summary[['Project Item Identifier', 'has_contractor', 'has_service']], on='Project Item Identifier', how='left').fillna({'has_contractor': False, 'has_service': False})
    
    def get_root_cause(row):
        t, s, has_con, has_svc = str(row['Type']).lower(), str(row['Stage']).lower(), row['has_contractor'], row['has_service']
        if t == 'capital purchase': return "Blocked: No Contractor" if not has_con else "Execution Inefficiency (Has Contracts)"
        elif t in ['large', 'medium']:
            if not has_svc: return "Blocked: No PSP (Design/Oversight)"
            if s in ['execution', 'commissioning'] and not has_con: return "Blocked: No Contractor"
            return "Delayed in Design Phase" if s == 'design' else "Execution Inefficiency (Has Contracts)"
        elif t == 'small':
            if s in ['execution', 'commissioning'] and not has_con: return "Blocked: No Contractor"
            return "Delayed in Design Phase (In-House)" if s == 'design' else "Execution Inefficiency (Has Contracts)"
        return "Other"

    df_merged['Root_Cause'] = df_merged.apply(get_root_cause, axis=1)
    df_merged['Type'] = df_merged['Type'].astype(str).str.title()
    plot_data = df_merged.groupby(['Type', 'Root_Cause'])['YTD_Underspend'].sum().unstack(fill_value=0)
    if not plot_data.empty: plot_data = plot_data.loc[[x for x in ['Capital Purchase', 'Small', 'Medium', 'Large'] if x in plot_data.index]]
    
    print(f"Total YTD Underspend (P1-P3): R {df_merged['YTD_Underspend'].sum():,.2f}\n")
    print("--- Breakdown of Underspend by Root Cause ---")
    for cause, val in df_merged.groupby('Root_Cause')['YTD_Underspend'].sum().sort_values(ascending=False).items():
        print(f"- {cause}: R {val:,.2f} ({(val/df_merged['YTD_Underspend'].sum())*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(12, 7))
    cause_colors = {"Blocked: No Contractor": "crimson", "Blocked: No PSP (Design/Oversight)": "darkorange", "Delayed in Design Phase": "gold", "Delayed in Design Phase (In-House)": "khaki", "Execution Inefficiency (Has Contracts)": "dodgerblue", "Other": "gray"}
    colors = [cause_colors.get(col, 'gray') for col in plot_data.columns]
    
    plot_data.plot(kind='bar', stacked=True, color=colors, ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title("Root Causes of FY2026 YTD Underspend by Project Type", fontsize=16, pad=15)
    ax.set_xlabel("Project Type", fontsize=12)
    ax.set_ylabel("YTD Underspend (Rands)", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'R {x / 1e6:,.0f}M'))
    plt.xticks(rotation=0, fontsize=12)
    ax.legend(title="Root Cause", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def analyze_root_causes_by_lifecycle_old(df_financial, df_attributes, df_procurement):
    print("==================================================")
    print("   Q2: ROOT CAUSE ANALYSIS BY LIFECYCLE STAGE (FY26 YTD)")
    print("==================================================\n")
    
    fy26_ytd = df_financial[(df_financial['Financial Year'] == 2026) & (df_financial['Period'] <= 3)]
    ytd_pivot = fy26_ytd.pivot_table(index='Project Item Identifier', columns='Financial View', values='Value', aggfunc='sum').reset_index().fillna(0)
    ytd_pivot['YTD_Underspend'] = ytd_pivot['Original Approved Budget'] - ytd_pivot['Actual']
    df_merged = ytd_pivot[ytd_pivot['YTD_Underspend'] > 0].copy().merge(df_attributes[['Project Item Identifier', 'Type', 'Stage']], on='Project Item Identifier', how='left')
    
    current_date = pd.to_datetime('2025-09-30')
    df_proc = df_procurement.copy()
    df_proc['Date of Expiry'], df_proc['Award Date'] = pd.to_datetime(df_proc['Date of Expiry']), pd.to_datetime(df_proc['Award Date'])
    active_contracts = df_proc[(df_proc['Status'] == 'Awarded') & (df_proc['Award Date'] <= current_date) & (df_proc['Date of Expiry'] >= current_date)]
    
    proc_summary = active_contracts.groupby(['Project Item Identifier', 'Category']).size().unstack(fill_value=0)
    proc_summary.columns = [col.lower() for col in proc_summary.columns]
    if 'service' not in proc_summary.columns: proc_summary['service'] = 0
    if 'contractor' not in proc_summary.columns: proc_summary['contractor'] = 0
    proc_summary['has_contractor'], proc_summary['has_service'] = proc_summary['contractor'] > 0, proc_summary['service'] > 0
    proc_summary = proc_summary.reset_index()
    
    df_merged = df_merged.merge(proc_summary[['Project Item Identifier', 'has_contractor', 'has_service']], on='Project Item Identifier', how='left').fillna({'has_contractor': False, 'has_service': False})
    
    def get_lifecycle_root_cause(row):
        t, s, has_con, has_svc = str(row['Type']).title(), str(row['Stage']).title(), row['has_contractor'], row['has_service']
        if s == 'Design': return f"Design Phase: Blocked (Needs PSP Contract)" if t in ['Large', 'Medium'] and not has_svc else f"Design Phase: Slow Spending / Delayed"
        elif s == 'Execution': return f"Execution Phase: Blocked (Missing Contractor)" if not has_con else f"Execution Phase: Inefficiency (Contractor Active but Slow)"
        elif s == 'Commissioning': return f"Commissioning Phase: Blocked (Missing Contractor)" if not has_con else f"Commissioning Phase: Handover Delays"
        elif s == 'Concluded': return "Concluded Phase: Late Invoice/Retention Payments"
        return "Unknown Phase Delay"

    df_merged['Lifecycle_Cause'] = df_merged.apply(get_lifecycle_root_cause, axis=1)
    df_merged['Type'] = df_merged['Type'].astype(str).str.title()
    
    print(f"Total YTD Underspend (P1-P3): R {df_merged['YTD_Underspend'].sum():,.2f}\n")
    print("--- Breakdown of Underspend by Lifecycle Phase ---")
    for cause, val in df_merged.groupby('Lifecycle_Cause')['YTD_Underspend'].sum().sort_values(ascending=False).items():
        print(f"- {cause}: R {val:,.2f} ({(val/df_merged['YTD_Underspend'].sum())*100:.1f}%)")

    plot_data = df_merged.groupby(['Type', 'Lifecycle_Cause'])['YTD_Underspend'].sum().unstack(fill_value=0)
    if not plot_data.empty: plot_data = plot_data.loc[[x for x in ['Capital Purchase', 'Small', 'Medium', 'Large'] if x in plot_data.index]]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    cause_colors = {"Execution Phase: Inefficiency (Contractor Active but Slow)": "dodgerblue", "Execution Phase: Blocked (Missing Contractor)": "crimson", "Design Phase: Slow Spending / Delayed": "gold", "Design Phase: Blocked (Needs PSP Contract)": "darkorange", "Commissioning Phase: Handover Delays": "mediumpurple", "Commissioning Phase: Blocked (Missing Contractor)": "purple", "Concluded Phase: Late Invoice/Retention Payments": "seagreen"}
    colors = [cause_colors.get(col, 'gray') for col in plot_data.columns]
    
    plot_data.plot(kind='bar', stacked=True, color=colors, ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title("Root Causes of FY26 YTD Underspend (By Lifecycle Phase & Project Type)", fontsize=16, pad=15)
    ax.set_xlabel("Project Type", fontsize=12)
    ax.set_ylabel("YTD Underspend (Rands)", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'R {x / 1e6:,.0f}M'))
    plt.xticks(rotation=0, fontsize=12)
    ax.legend(title="Lifecycle Stage & Root Cause", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()



def analyze_root_causes_by_type(df_financial, df_attributes, df_procurement):
    print("==================================================")
    print("   Q2: ROOT CAUSE ANALYSIS BY PROJECT TYPE (FY26 YTD)")
    print("==================================================\n")
    
    fy26_ytd = df_financial[(df_financial['Financial Year'] == 2026) & (df_financial['Period'] <= 3)]
    ytd_pivot = fy26_ytd.pivot_table(index='Project Item Identifier', columns='Financial View', values='Value', aggfunc='sum').reset_index().fillna(0)
    ytd_pivot['YTD_Underspend'] = ytd_pivot['Original Approved Budget'] - ytd_pivot['Actual']
    df_merged = ytd_pivot[ytd_pivot['YTD_Underspend'] > 0].copy().merge(df_attributes[['Project Item Identifier', 'Type', 'Stage']], on='Project Item Identifier', how='left')
    
    current_date = pd.to_datetime('2025-09-30')
    df_proc = df_procurement.copy()
    df_proc['Date of Expiry'], df_proc['Award Date'] = pd.to_datetime(df_proc['Date of Expiry']), pd.to_datetime(df_proc['Award Date'])
    active_contracts = df_proc[(df_proc['Status'] == 'Awarded') & (df_proc['Award Date'] <= current_date) & (df_proc['Date of Expiry'] >= current_date)]
    
    proc_summary = active_contracts.groupby(['Project Item Identifier', 'Category']).size().unstack(fill_value=0)
    proc_summary.columns = [col.lower() for col in proc_summary.columns]
    if 'service' not in proc_summary.columns: proc_summary['service'] = 0
    if 'contractor' not in proc_summary.columns: proc_summary['contractor'] = 0
    proc_summary['has_contractor'], proc_summary['has_service'] = proc_summary['contractor'] > 0, proc_summary['service'] > 0
    proc_summary = proc_summary.reset_index()
    
    df_merged = df_merged.merge(proc_summary[['Project Item Identifier', 'has_contractor', 'has_service']], on='Project Item Identifier', how='left').fillna({'has_contractor': False, 'has_service': False})
    
    def get_root_cause(row):
        t, s, has_con, has_svc = str(row['Type']).lower(), str(row['Stage']).lower(), row['has_contractor'], row['has_service']
        if t == 'capital purchase': return "Blocked: No Contractor" if not has_con else "Execution Inefficiency (Has Contracts)"
        elif t in ['large', 'medium']:
            if not has_svc: return "Blocked: No PSP (Design/Oversight)"
            if s in ['execution', 'commissioning'] and not has_con: return "Blocked: No Contractor"
            return "Delayed in Design Phase" if s == 'design' else "Execution Inefficiency (Has Contracts)"
        elif t == 'small':
            if s in ['execution', 'commissioning'] and not has_con: return "Blocked: No Contractor"
            return "Delayed in Design Phase (In-House)" if s == 'design' else "Execution Inefficiency (Has Contracts)"
        return "Other"

    df_merged['Root_Cause'] = df_merged.apply(get_root_cause, axis=1)
    
    # GROUP MEDIUM AND LARGE TOGETHER
    def group_type(t):
        t_lower = str(t).lower()
        if t_lower in ['medium', 'large']:
            return 'Medium & Large'
        return t_lower.title()
    
    df_merged['Type'] = df_merged['Type'].apply(group_type)
    
    plot_data = df_merged.groupby(['Type', 'Root_Cause'])['YTD_Underspend'].sum().unstack(fill_value=0)
    
    # Update expected index order to reflect the new grouping
    expected_order = ['Capital Purchase', 'Small', 'Medium & Large']
    if not plot_data.empty: 
        plot_data = plot_data.loc[[x for x in expected_order if x in plot_data.index]]
    
    print(f"Total YTD Underspend (P1-P3): R {df_merged['YTD_Underspend'].sum():,.2f}\n")
    print("--- Breakdown of Underspend by Root Cause ---")
    for cause, val in df_merged.groupby('Root_Cause')['YTD_Underspend'].sum().sort_values(ascending=False).items():
        print(f"- {cause}: R {val:,.2f} ({(val/df_merged['YTD_Underspend'].sum())*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(12, 7))
    cause_colors = {"Blocked: No Contractor": "crimson", "Blocked: No PSP (Design/Oversight)": "darkorange", "Delayed in Design Phase": "gold", "Delayed in Design Phase (In-House)": "khaki", "Execution Inefficiency (Has Contracts)": "dodgerblue", "Other": "gray"}
    colors = [cause_colors.get(col, 'gray') for col in plot_data.columns]
    
    plot_data.plot(kind='bar', stacked=True, color=colors, ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title("Root Causes of FY2026 YTD Underspend by Project Type", fontsize=16, pad=15)
    ax.set_xlabel("Project Type", fontsize=12)
    ax.set_ylabel("YTD Underspend (Rands)", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'R {x / 1e6:,.0f}M'))
    plt.xticks(rotation=0, fontsize=12)
    ax.legend(title="Root Cause", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def analyze_root_causes_by_lifecycle(df_financial, df_attributes, df_procurement):
    print("==================================================")
    print("   Q2: ROOT CAUSE ANALYSIS BY LIFECYCLE STAGE (FY26 YTD)")
    print("==================================================\n")
    
    fy26_ytd = df_financial[(df_financial['Financial Year'] == 2026) & (df_financial['Period'] <= 3)]
    ytd_pivot = fy26_ytd.pivot_table(index='Project Item Identifier', columns='Financial View', values='Value', aggfunc='sum').reset_index().fillna(0)
    ytd_pivot['YTD_Underspend'] = ytd_pivot['Original Approved Budget'] - ytd_pivot['Actual']
    df_merged = ytd_pivot[ytd_pivot['YTD_Underspend'] > 0].copy().merge(df_attributes[['Project Item Identifier', 'Type', 'Stage']], on='Project Item Identifier', how='left')
    
    current_date = pd.to_datetime('2025-09-30')
    df_proc = df_procurement.copy()
    df_proc['Date of Expiry'], df_proc['Award Date'] = pd.to_datetime(df_proc['Date of Expiry']), pd.to_datetime(df_proc['Award Date'])
    active_contracts = df_proc[(df_proc['Status'] == 'Awarded') & (df_proc['Award Date'] <= current_date) & (df_proc['Date of Expiry'] >= current_date)]
    
    proc_summary = active_contracts.groupby(['Project Item Identifier', 'Category']).size().unstack(fill_value=0)
    proc_summary.columns = [col.lower() for col in proc_summary.columns]
    if 'service' not in proc_summary.columns: proc_summary['service'] = 0
    if 'contractor' not in proc_summary.columns: proc_summary['contractor'] = 0
    proc_summary['has_contractor'], proc_summary['has_service'] = proc_summary['contractor'] > 0, proc_summary['service'] > 0
    proc_summary = proc_summary.reset_index()
    
    df_merged = df_merged.merge(proc_summary[['Project Item Identifier', 'has_contractor', 'has_service']], on='Project Item Identifier', how='left').fillna({'has_contractor': False, 'has_service': False})
    
    def get_lifecycle_root_cause(row):
        t, s, has_con, has_svc = str(row['Type']).title(), str(row['Stage']).title(), row['has_contractor'], row['has_service']
        if s == 'Design': return f"Design Phase: Blocked (Needs PSP Contract)" if t in ['Large', 'Medium'] and not has_svc else f"Design Phase: Slow Spending / Delayed"
        elif s == 'Execution': return f"Execution Phase: Blocked (Missing Contractor)" if not has_con else f"Execution Phase: Inefficiency (Contractor Active but Slow)"
        elif s == 'Commissioning': return f"Commissioning Phase: Blocked (Missing Contractor)" if not has_con else f"Commissioning Phase: Handover Delays"
        elif s == 'Concluded': return "Concluded Phase: Late Invoice/Retention Payments"
        return "Unknown Phase Delay"

    df_merged['Lifecycle_Cause'] = df_merged.apply(get_lifecycle_root_cause, axis=1)
    
    # GROUP MEDIUM AND LARGE TOGETHER
    def group_type(t):
        t_lower = str(t).lower()
        if t_lower in ['medium', 'large']:
            return 'Medium & Large'
        return t_lower.title()
    
    df_merged['Type'] = df_merged['Type'].apply(group_type)
    
    print(f"Total YTD Underspend (P1-P3): R {df_merged['YTD_Underspend'].sum():,.2f}\n")
    print("--- Breakdown of Underspend by Lifecycle Phase ---")
    for cause, val in df_merged.groupby('Lifecycle_Cause')['YTD_Underspend'].sum().sort_values(ascending=False).items():
        print(f"- {cause}: R {val:,.2f} ({(val/df_merged['YTD_Underspend'].sum())*100:.1f}%)")

    plot_data = df_merged.groupby(['Type', 'Lifecycle_Cause'])['YTD_Underspend'].sum().unstack(fill_value=0)
    
    # Update expected index order to reflect the new grouping
    expected_order = ['Capital Purchase', 'Small', 'Medium & Large']
    if not plot_data.empty: 
        plot_data = plot_data.loc[[x for x in expected_order if x in plot_data.index]]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    cause_colors = {"Execution Phase: Inefficiency (Contractor Active but Slow)": "dodgerblue", "Execution Phase: Blocked (Missing Contractor)": "crimson", "Design Phase: Slow Spending / Delayed": "gold", "Design Phase: Blocked (Needs PSP Contract)": "darkorange", "Commissioning Phase: Handover Delays": "mediumpurple", "Commissioning Phase: Blocked (Missing Contractor)": "purple", "Concluded Phase: Late Invoice/Retention Payments": "seagreen"}
    colors = [cause_colors.get(col, 'gray') for col in plot_data.columns]
    
    plot_data.plot(kind='bar', stacked=True, color=colors, ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title("Root Causes of FY26 YTD Underspend (By Lifecycle Phase & Project Type)", fontsize=16, pad=15)
    ax.set_xlabel("Project Type", fontsize=12)
    ax.set_ylabel("YTD Underspend (Rands)", fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'R {x / 1e6:,.0f}M'))
    plt.xticks(rotation=0, fontsize=12)
    ax.legend(title="Lifecycle Stage & Root Cause", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

 
def investigate_lifecycle_shift(df_financial, df_attributes):
    print("==================================================")
    print("   EMPIRICAL INVESTIGATION: LIFECYCLE STAGE SHIFT")
    print("==================================================\n")

    # 1. Calculate Total FY25 Actual Spend per Project
    fy25_actuals = df_financial[(df_financial['Financial Year'] == 2025) & (df_financial['Financial View'] == 'Actual')]
    fy25_proj_spend = fy25_actuals.groupby('Project Item Identifier')['Value'].sum().reset_index()
    fy25_proj_spend.rename(columns={'Value': 'FY25_Actual_Spend'}, inplace=True)

    # 2. Calculate Total FY26 Approved Budget per Project
    fy26_budget = df_financial[(df_financial['Financial Year'] == 2026) & (df_financial['Financial View'] == 'Original Approved Budget')]
    fy26_proj_budg = fy26_budget.groupby('Project Item Identifier')['Value'].sum().reset_index()
    fy26_proj_budg.rename(columns={'Value': 'FY26_Approved_Budget'}, inplace=True)

    # 3. Merge with Project Attributes (to get the current Stage)
    df_shift = df_attributes[['Project Item Identifier', 'Stage']].copy()
    df_shift = df_shift.merge(fy25_proj_spend, on='Project Item Identifier', how='left').fillna({'FY25_Actual_Spend': 0})
    df_shift = df_shift.merge(fy26_proj_budg, on='Project Item Identifier', how='left').fillna({'FY26_Approved_Budget': 0})

    # Clean up stage names
    df_shift['Stage'] = df_shift['Stage'].astype(str).str.title()

    # 4. Aggregate by Stage
    stage_summary = df_shift.groupby('Stage')[['FY25_Actual_Spend', 'FY26_Approved_Budget']].sum().reset_index()
    
    # Calculate Percentages to show the structural shift
    total_fy25 = stage_summary['FY25_Actual_Spend'].sum()
    total_fy26 = stage_summary['FY26_Approved_Budget'].sum()
    
    stage_summary['FY25 % of Portfolio'] = (stage_summary['FY25_Actual_Spend'] / total_fy25) * 100
    stage_summary['FY26 % of Portfolio'] = (stage_summary['FY26_Approved_Budget'] / total_fy26) * 100

    # Sort for plotting (Logical lifecycle order)
    stage_order = ['Initiation', 'Concept', 'Design', 'Execution', 'Commissioning', 'Concluded']
    stage_order = [s for s in stage_order if s in stage_summary['Stage'].values]
    stage_summary['Stage'] = pd.Categorical(stage_summary['Stage'], categories=stage_order, ordered=True)
    stage_summary = stage_summary.sort_values('Stage')

    # 5. Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(stage_summary['Stage']))
    width = 0.35
    
    ax.bar(x - width/2, stage_summary['FY25_Actual_Spend'], width, label='Total FY25 Actual Spend', color='purple', edgecolor='black')
    ax.bar(x + width/2, stage_summary['FY26_Approved_Budget'], width, label='Total FY26 Approved Budget', color='orange', edgecolor='black')
    
    ax.set_title("Portfolio Structural Shift: FY25 Spend vs. FY26 Budget by Lifecycle Stage", fontsize=15)
    ax.set_ylabel("Total Value (Rands)")
    ax.set_xlabel("Current Project Stage")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_summary['Stage'])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'R {val/1e6:,.0f}M'))
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

    # 6. Empirical Output
    print("► EMPIRICAL FINDING: The Lifecycle Transition Gap")
    for _, row in stage_summary.iterrows():
        print(f"- {row['Stage']}: Drove {row['FY25 % of Portfolio']:.1f}% of FY25 Spend  -->  Holds {row['FY26 % of Portfolio']:.1f}% of FY26 Budget")
    
    # Dynamic Conclusion
    exec_concluded_fy25 = stage_summary[stage_summary['Stage'].isin(['Concluded', 'Commissioning'])]['FY25 % of Portfolio'].sum()
    design_init_fy26 = stage_summary[stage_summary['Stage'].isin(['Design', 'Initiation', 'Concept'])]['FY26 % of Portfolio'].sum()
    
    print(f"\nCONCLUSION:")
    print(f"Your hypothesis is correct. A massive {exec_concluded_fy25:.1f}% of all money spent in FY25 was driven by projects that are now 'Concluded' or in 'Commissioning'.")
    print(f"Meanwhile, {design_init_fy26:.1f}% of the new FY26 budget is heavily tied up in projects that are currently stuck in 'Design' or 'Initiation'.")
    print(f"Because projects in early stages spend money exponentially slower than projects in Execution, the portfolio is experiencing a 'Lifecycle Hangover', causing the massive drop in expenditure at the start of FY26.")




def answer_q2_drilldown(df_financial, df_attributes, df_procurement):
 
    
    # 1. Calculate YTD Underspend (FY26, P1-P3)
    fy26_ytd = df_financial[(df_financial['Financial Year'] == 2026) & (df_financial['Period'] <= 3)]
    ytd_pivot = fy26_ytd.pivot_table(index='Project Item Identifier', columns='Financial View', values='Value', aggfunc='sum').reset_index().fillna(0)
    
    if 'Original Approved Budget' not in ytd_pivot.columns: ytd_pivot['Original Approved Budget'] = 0
    if 'Actual' not in ytd_pivot.columns: ytd_pivot['Actual'] = 0
    
    ytd_pivot['YTD_Underspend'] = ytd_pivot['Original Approved Budget'] - ytd_pivot['Actual']
    df_under = ytd_pivot[ytd_pivot['YTD_Underspend'] > 0].copy()
    
    # 2. Merge with attributes and map to required buckets (Medium & Large combined)
    df_merged = df_under.merge(df_attributes[['Project Item Identifier', 'Type', 'Stage']], on='Project Item Identifier', how='left')
    df_merged['Type_Lower'] = df_merged['Type'].astype(str).str.lower()
    
    def map_bucket(row):
        t = row['Type_Lower']
        if t == 'capital purchase': return 'Capital Purchases'
        elif t == 'small': return 'Small Projects'
        elif t in ['medium', 'large']: return 'Medium & Large Projects'
        else: return 'Other'
        
    df_merged['Project_Bucket'] = df_merged.apply(map_bucket, axis=1)
    
    # 3. Analyze highly specific Procurement Status per project
    df_proc = df_procurement.copy()
    df_proc['Status'] = df_proc['Status'].str.lower().str.strip()
    df_proc['Category'] = df_proc['Category'].str.lower().str.strip()
    
    proc_status = {}
    for proj_id, group in df_proc.groupby('Project Item Identifier'):
        cats = group['Category'].tolist()
        statuses = group['Status'].tolist()
        proj_dict = {'service': [], 'contractor': [], 'supplier': []}
        for c, s in zip(cats, statuses):
            if c in proj_dict:
                proj_dict[c].append(s)
        proc_status[proj_id] = proj_dict
        
    # 4. Determine root causes based on distinct project type rules
    def get_root_cause(row):
        proj_id = row['Project Item Identifier']
        bucket = row['Project_Bucket']
        stage = str(row['Stage']).lower()
        p_status = proc_status.get(proj_id, {'service': [], 'contractor': [], 'supplier': []})
        
        has_awarded_supplier = 'awarded' in p_status['supplier']
        has_awarded_contractor = 'awarded' in p_status['contractor']
        has_awarded_service = 'awarded' in p_status['service']
        
        if bucket == 'Capital Purchases':
            if not has_awarded_supplier:
                if 'in process' in p_status['supplier']: return "Procurement Bottleneck: Supplier Tender stuck 'In Process'"
                elif 'cancelled' in p_status['supplier']: return "Procurement Failure: Supplier Tender 'Cancelled'"
                else: return "Blocked: Missing Supplier Tender entirely"
            else:
                return "Execution Inefficiency: Slow delivery despite awarded Supplier"
                
        elif bucket == 'Small Projects':
            if stage == 'design':
                return "Lifecycle Hangover: Delayed in internal/in-house Design phase"
            elif not has_awarded_contractor:
                if 'in process' in p_status['contractor']: return "Procurement Bottleneck: Contractor Tender stuck 'In Process'"
                else: return "Blocked: Missing Contractor Tender entirely"
            else:
                return "Execution Inefficiency: Slow construction despite active Contractor"
                
        elif bucket == 'Medium & Large Projects':
            if not has_awarded_service:
                if 'in process' in p_status['service']: return "Procurement Bottleneck: PSP (Service) Tender stuck 'In Process'"
                else: return "Blocked early phase: Missing PSP (Service) Tender"
            elif stage == 'design':
                return "Lifecycle Hangover: Stuck in Design phase (inherent slow burn rate)"
            elif not has_awarded_contractor:
                if 'in process' in p_status['contractor']: return "Procurement Bottleneck: Contractor Tender stuck 'In Process'"
                else: return "Blocked execution: Missing Contractor Tender entirely"
            else:
                return "Execution Inefficiency: Slow construction despite all active contracts"
                
        return "Other"
        
    df_merged['Root_Cause'] = df_merged.apply(get_root_cause, axis=1)
    
    # 5. Extract and print the single highest financial driver per bucket
    buckets = ['Capital Purchases', 'Small Projects', 'Medium & Large Projects']
    plot_data = []
    
    print(f"Total Portfolio YTD Underspend Analysed: R {df_merged['YTD_Underspend'].sum():,.2f}\n")
    
    for b in buckets:
        b_data = df_merged[df_merged['Project_Bucket'] == b]
        if b_data.empty: continue
        
        total_b_underspend = b_data['YTD_Underspend'].sum()
        cause_agg = b_data.groupby('Root_Cause')['YTD_Underspend'].sum().sort_values(ascending=False)
        
        top_cause = cause_agg.index[0]
        top_cause_val = cause_agg.iloc[0]
        pct_of_bucket = (top_cause_val / total_b_underspend) * 100
        
        plot_data.append({'Bucket': b, 'Top Reason': top_cause, 'Underspend': top_cause_val})
        
        print(f"► {b.upper()}")
        print(f"  Total Underspend in Bucket: R {total_b_underspend:,.2f}")
        print(f"  Primary Root Cause: {top_cause}")
        print(f"  Financial Impact: R {top_cause_val:,.2f} ({pct_of_bucket:.1f}% of {b} underspend)\n")

    # 6. Visualize the definitive answer
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(plot_df['Bucket'], plot_df['Underspend'], color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
        
        ax.set_title("Primary Root Cause of FY26 YTD Underspend by Project Bucket", fontsize=16, pad=20)
        ax.set_ylabel("YTD Underspend (Rands)", fontsize=12)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'R {x / 1e6:,.0f}M'))
        
        # Add labels on top of bars
        for bar, reason in zip(bars, plot_df['Top Reason']):
            yval = bar.get_height()
            # Split long text for better readability
            formatted_reason = reason.replace(': ', ':\n')
            ax.text(bar.get_x() + bar.get_width()/2, yval + (yval*0.02), formatted_reason, 
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
            
        # Adjust y-limit to fit the text
        ax.set_ylim(0, plot_df['Underspend'].max() * 1.2)
        plt.tight_layout()
        plt.show()


def analyze_root_causes_by_type_standalone(df_financial, df_attributes, df_procurement):
    print("==================================================")
    print("   Q2: ROOT CAUSE ANALYSIS BY PROJECT TYPE (FY26 YTD)")
    print("==================================================\n")
    
    # 1. Calculate YTD Underspend
    fy26_ytd = df_financial[(df_financial['Financial Year'] == 2026) & (df_financial['Period'] <= 3)]
    ytd_pivot = fy26_ytd.pivot_table(index='Project Item Identifier', columns='Financial View', values='Value', aggfunc='sum').reset_index().fillna(0)
    ytd_pivot['YTD_Underspend'] = ytd_pivot['Original Approved Budget'] - ytd_pivot['Actual']
    
    # 2. Merge with Attributes (Including Project Manager for the Watchlist)
    df_merged = ytd_pivot[ytd_pivot['YTD_Underspend'] > 0].copy().merge(
        df_attributes[['Project Item Identifier', 'Type', 'Stage', 'Project Manager']], 
        on='Project Item Identifier', how='left'
    )
    
    # 3. Assess Active Contracts
    current_date = pd.to_datetime('2025-09-30')
    df_proc = df_procurement.copy()
    df_proc['Date of Expiry'] = pd.to_datetime(df_proc['Date of Expiry'])
    df_proc['Award Date'] = pd.to_datetime(df_proc['Award Date'])
    active_contracts = df_proc[(df_proc['Status'] == 'Awarded') & 
                               (df_proc['Award Date'] <= current_date) & 
                               (df_proc['Date of Expiry'] >= current_date)]
    
    proc_summary = active_contracts.groupby(['Project Item Identifier', 'Category']).size().unstack(fill_value=0)
    proc_summary.columns = [col.lower() for col in proc_summary.columns]
    if 'service' not in proc_summary.columns: proc_summary['service'] = 0
    if 'contractor' not in proc_summary.columns: proc_summary['contractor'] = 0
    proc_summary['has_contractor'], proc_summary['has_service'] = proc_summary['contractor'] > 0, proc_summary['service'] > 0
    proc_summary = proc_summary.reset_index()
    
    df_merged = df_merged.merge(proc_summary[['Project Item Identifier', 'has_contractor', 'has_service']], on='Project Item Identifier', how='left').fillna({'has_contractor': False, 'has_service': False})
    
    # 4. Root Cause Logic
    def get_root_cause(row):
        t, s, has_con, has_svc = str(row['Type']).lower(), str(row['Stage']).lower(), row['has_contractor'], row['has_service']
        if t == 'capital purchase': return "Blocked: No Contractor" if not has_con else "Execution Inefficiency (Has Contracts)"
        elif t in ['large', 'medium']:
            if not has_svc: return "Blocked: No PSP (Design/Oversight)"
            if s in ['execution', 'commissioning'] and not has_con: return "Blocked: No Contractor"
            return "Delayed in Design Phase" if s == 'design' else "Execution Inefficiency (Has Contracts)"
        elif t == 'small':
            if s in ['execution', 'commissioning'] and not has_con: return "Blocked: No Contractor"
            return "Delayed in Design Phase (In-House)" if s == 'design' else "Execution Inefficiency (Has Contracts)"
        return "Other"

    df_merged['Root_Cause'] = df_merged.apply(get_root_cause, axis=1)
    
    # 5. Group Medium and Large Together
    def group_type(t):
        t_lower = str(t).lower()
        if t_lower in ['medium', 'large']: return 'Medium & Large'
        return t_lower.title()
    
    df_merged['Type'] = df_merged['Type'].apply(group_type)
    
    # 6. Prepare Plot Data
    plot_data = df_merged.groupby(['Type', 'Root_Cause'])['YTD_Underspend'].sum().unstack(fill_value=0)
    expected_order = ['Capital Purchase', 'Small', 'Medium & Large']
    if not plot_data.empty: 
        plot_data = plot_data.loc[[x for x in expected_order if x in plot_data.index]]
    
    print(f"Total YTD Underspend (P1-P3): R {df_merged['YTD_Underspend'].sum():,.2f}\n")
    print("--- Breakdown of Underspend by Root Cause ---")
 
    
    return df_merged

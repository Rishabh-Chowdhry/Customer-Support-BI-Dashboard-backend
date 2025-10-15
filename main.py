from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import uvicorn

app = FastAPI(title="Customer Support BI Dashboard API")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- MOCK DATA GENERATOR ----------
def generate_mock_data():
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2025-10-15', freq='h')
    regions = ['Lahore', 'Karachi', 'Islamabad', 'Peshawar', 'Quetta']
    teams   = ['Telecom', 'Banking', 'Retail', 'Insurance', 'Utilities']

    region_factors = {'Lahore': 1.5, 'Karachi': 1.8, 'Islamabad': 0.8,
                      'Peshawar': 0.6, 'Quetta': 0.3}

    team_patterns = {
        'Telecom':   {'base_calls': 120, 'csat_base': 3.8, 'aht_base': 8.5, 'churn_base': 0.025},
        'Banking':   {'base_calls': 95,  'csat_base': 4.1, 'aht_base': 10.2, 'churn_base': 0.015},
        'Retail':    {'base_calls': 75,  'csat_base': 3.9, 'aht_base': 7.8, 'churn_base': 0.020},
        'Insurance': {'base_calls': 55,  'csat_base': 3.7, 'aht_base': 14.5, 'churn_base': 0.018},
        'Utilities': {'base_calls': 40,  'csat_base': 3.6, 'aht_base': 9.8, 'churn_base': 0.028}
    }

    data = []
    for date in dates:
        year_progress   = (date - pd.Timestamp('2020-01-01')).total_seconds() / (5 * 365.25 * 24 * 3600)
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
        trend_factor    = 1 + 0.1 * year_progress
        weekend_factor  = 0.6 if date.weekday() >= 5 else 1.0
        month_end_factor = 1.2 if date.day >= 25 else 1.0
        hourly_factor   = 1/24
        time_of_day_factor = 1.0 + 0.1 * np.sin(2 * np.pi * (date.hour - 9) / 16)

        for region in regions:
            region_factor = region_factors[region]
            for team in teams:
                cfg = team_patterns[team]

                total_calls = np.random.poisson(
                    cfg['base_calls'] * region_factor *
                    seasonal_factor * trend_factor *
                    weekend_factor * month_end_factor *
                    hourly_factor
                )
                total_calls = max(1, total_calls)

                workload_factor = min(1.0, total_calls / (cfg['base_calls'] * 1.2))
                csat_noise = np.random.normal(0, 0.15)
                csat = cfg['csat_base'] * workload_factor * time_of_day_factor + csat_noise
                csat = max(2.8, min(4.8, csat))

                complexity_factor = 1.0 + 0.2 * np.random.beta(2, 3)
                experience_factor = 0.85 + 0.15 * year_progress
                time_pressure = 1.0 + 0.1 * (total_calls / cfg['base_calls'] - 1)
                aht_noise = np.random.normal(0, 0.5)
                aht = cfg['aht_base'] * complexity_factor / experience_factor * time_pressure + aht_noise
                aht = max(4, min(25, aht))

                fcr_base = 0.68 + 0.12 * year_progress
                fcr_noise = np.random.beta(3, 2) - 0.5
                fcr = fcr_base + fcr_noise * 0.15
                fcr = max(0.35, min(0.92, fcr))

                csat_impact = (5 - csat) / 2.5
                aht_impact = max(0, (aht - 10) / 15)
                fcr_impact = (1 - fcr) * 0.3
                churn_noise = np.random.beta(0.8, 8) * 0.005
                churn = cfg['churn_base'] * (1 + csat_impact + aht_impact + fcr_impact) + churn_noise
                churn = min(churn, 0.05)

                base_occupancy = 0.72
                volume_factor = min(1.2, total_calls / (cfg['base_calls'] * 0.8))
                time_factor = 0.9 + 0.2 * np.sin(2 * np.pi * (date.hour - 8) / 16)
                occupancy_noise = np.random.normal(0, 0.05)
                occupancy = base_occupancy * volume_factor * time_factor + occupancy_noise
                occupancy = max(0.45, min(0.98, occupancy))

                data.append({
                    'date': date,
                    'region': region,
                    'team': team,
                    'total_calls': int(total_calls),
                    'csat': round(csat, 2),
                    'aht': round(aht, 2),
                    'fcr': round(fcr, 2),
                    'churn': round(churn, 4),
                    'occupancy': round(occupancy, 2)
                })
    return pd.DataFrame(data)

# ---------- GLOBAL DATA & CHURN MODEL ----------
df = generate_mock_data()

def train_churn_model():
    features = ['csat', 'aht', 'fcr', 'occupancy', 'total_calls']
    X = df[features]
    y = df['churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42)
    model.fit(X_train, y_train)
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, model.predict(X_test))
    print(f"Churn Model MSE: {mse:.6f}")
    return model

churn_model = train_churn_model()

# ---------- API ENDPOINTS ----------
@app.get("/")
def root():
    return {"message": "Customer-Support BI Dashboard API is running 🚀"}

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/api/overview")
def get_overview():
    latest = df[df['date'] == df['date'].max()]
    return {
        'total_calls': int(latest['total_calls'].sum()),
        'csat': round(latest['csat'].mean(), 2),
        'aht': round(latest['aht'].mean(), 2),
        'fcr': round(latest['fcr'].mean(), 2),
        'churn': round(latest['churn'].mean() * 100, 2),
        'occupancy': round(latest['occupancy'].mean() * 100, 2)
    }

@app.get("/api/trends")
def get_trends(days: int = 30):
    end_date = df['date'].max()
    start_date = end_date - timedelta(days=days)
    trends = df[df['date'] >= start_date].groupby('date').agg({
        'total_calls': 'sum', 'csat': 'mean', 'aht': 'mean',
        'fcr': 'mean', 'churn': 'mean', 'occupancy': 'mean'
    }).reset_index()
    return trends.to_dict('records')

@app.get("/api/region-breakdown")
def get_region_breakdown():
    latest = df[df['date'] == df['date'].max()]
    breakdown = latest.groupby('region').agg({
        'total_calls': 'sum', 'csat': 'mean', 'aht': 'mean',
        'fcr': 'mean', 'churn': 'mean'
    }).reset_index()
    return breakdown.to_dict('records')

@app.get("/api/diagnostics")
def get_diagnostics():
    latest = df[df['date'] == df['date'].max()]
    numeric = ['total_calls', 'csat', 'aht', 'fcr', 'churn', 'occupancy']
    corr = latest[numeric].corr()
    # Replace NaN values in correlation matrix with 0
    corr = corr.fillna(0)
    influencers = []
    for metric in ['csat', 'churn']:
        top_series = corr[metric].drop(metric).abs()
        if not top_series.empty:
            top = top_series.idxmax()
            influencers.append({
                'metric': metric,
                'influencer': top,
                'correlation': corr.loc[metric, top]
            })
        else:
            influencers.append({
                'metric': metric,
                'influencer': None,
                'correlation': 0
            })
    return {'correlations': corr.to_dict(), 'influencers': influencers}

@app.get("/api/drilldown/{region}")
def get_drilldown(region: str):
    latest = df[(df['date'] == df['date'].max()) & (df['region'] == region)]
    drill = latest.groupby('team').agg({
        'total_calls': 'sum', 'csat': 'mean', 'aht': 'mean',
        'fcr': 'mean', 'churn': 'mean'
    }).reset_index()
    return drill.to_dict('records')

@app.get("/api/forecast")
def get_forecast(days: int = 30):
    daily = df.groupby('date')['total_calls'].sum().reset_index()
    daily.columns = ['ds', 'y']
    model = Prophet()
    model.fit(daily)
    future = model.make_future_dataframe(periods=days)
    fc = model.predict(future)
    out = fc[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
    out.columns = ['date', 'forecast', 'lower', 'upper']
    return out.to_dict('records')

@app.get("/api/churn-prediction")
def get_churn_prediction():
    latest = df[df['date'] == df['date'].max()]
    features = ['csat', 'aht', 'fcr', 'occupancy', 'total_calls']
    X = latest[features]
    predicted_churn = churn_model.predict(X)
    # Handle NaN values by replacing with 0
    predicted_churn = np.nan_to_num(predicted_churn, nan=0.0)
    latest = latest.copy()
    latest['predicted_churn'] = predicted_churn
    summary = latest.groupby(['region', 'team']).agg({
        'predicted_churn': 'mean', 'total_calls': 'sum'
    }).reset_index()
    # Ensure no NaN values in the result
    summary = summary.fillna(0)
    return summary.to_dict('records')

@app.get("/api/prescriptive-insights")
def get_prescriptive_insights():
    latest = df[df['date'] == df['date'].max()]
    insights = []
    occ = latest['occupancy'].mean()
    if occ > 0.8:
        insights.append({'type': 'staffing', 'priority': 'high',
                         'message': f"⚠️ High occupancy ({occ:.1%}). Increase staffing 10-15%."})
    csat = latest['csat'].mean()
    if csat < 4.0:
        insights.append({'type': 'satisfaction', 'priority': 'high',
                         'message': f"📉 CSAT below target ({csat:.2f}/5). Train agents."})
    churn_by_reg = latest.groupby('region')['churn'].mean()
    if churn_by_reg.max() > 0.02:
        insights.append({'type': 'churn', 'priority': 'medium',
                         'message': f"⚠️ High churn in {churn_by_reg.idxmax()}. Run retention campaigns."})
    aht = latest['aht'].mean()
    if aht > 9:
        insights.append({'type': 'efficiency', 'priority': 'medium',
                         'message': f"⏱️ AHT high ({aht:.1f} min). Add efficiency measures."})
    return insights

# ---------- FILTERED & CHART HELPERS ----------
def _filter(df_in, region=None, team=None, start_date=None, end_date=None,
            min_calls=None, max_calls=None, latest_if_empty=True):
    df_out = df_in.copy()
    if region:      df_out = df_out[df_out['region'] == region]
    if team:        df_out = df_out[df_out['team'] == team]
    if start_date:  df_out = df_out[df_out['date'] >= pd.to_datetime(start_date)]
    if end_date:    df_out = df_out[df_out['date'] <= pd.to_datetime(end_date)]
    if min_calls is not None: df_out = df_out[df_out['total_calls'] >= min_calls]
    if max_calls is not None: df_out = df_out[df_out['total_calls'] <= max_calls]
    if latest_if_empty and not start_date and not end_date:
        df_out = df_out[df_out['date'] == df_out['date'].max()]
    return df_out

@app.get("/api/boxplot-data")
def get_boxplot_data(region: str = None, team: str = None,
                     start_date: str = None, end_date: str = None,
                     min_calls: int = None, max_calls: int = None):
    filtered = _filter(df, region, team, start_date, end_date, min_calls, max_calls)
    out = {}
    for metric in ['csat', 'aht', 'fcr', 'occupancy']:
        regional = []
        for reg in filtered['region'].unique():
            reg_data = filtered[filtered['region'] == reg]
            vals = reg_data[metric].dropna().values.tolist()
            if vals: regional.append({'region': reg, 'data': vals})
        out[metric] = regional
    return out

@app.get("/api/sunburst-data")
def get_sunburst_data(region: str = None, team: str = None,
                       start_date: str = None, end_date: str = None,
                       min_calls: int = None, max_calls: int = None):
    filtered = _filter(df, region, team, start_date, end_date, min_calls, max_calls)
    data = []
    for reg in filtered['region'].unique():
        reg_df = filtered[filtered['region'] == reg]
        reg_calls = int(reg_df['total_calls'].sum())
        data.append({'id': reg, 'parent': '', 'name': reg, 'value': reg_calls})
        for tm in reg_df['team'].unique():
            team_calls = int(reg_df[reg_df['team'] == tm]['total_calls'].sum())
            data.append({'id': f"{reg}-{tm}", 'parent': reg,
                         'name': tm, 'value': team_calls})
    return data

@app.get("/api/regression-data")
def get_regression_data(region: str = None, team: str = None,
                        start_date: str = None, end_date: str = None,
                        min_calls: int = None, max_calls: int = None):
    from scipy import stats
    filtered = _filter(df, region, team, start_date, end_date, min_calls, max_calls)
    # Drop rows with NaN in aht or csat
    filtered = filtered.dropna(subset=['aht', 'csat'])
    if len(filtered) < 2:
        return {'scatter_points': [], 'regression_line': {
            'x': [], 'y': [], 'slope': 0, 'intercept': 0, 'r_squared': 0}}
    x = filtered['aht'].values
    y = filtered['csat'].values
    slope, intercept, r, _, _ = stats.linregress(x, y)
    # Handle NaN values in regression results
    if np.isnan(slope) or np.isnan(intercept) or np.isnan(r):
        return {'scatter_points': filtered[['aht', 'csat']].to_dict('records'), 'regression_line': {
            'x': [], 'y': [], 'slope': 0, 'intercept': 0, 'r_squared': 0}}
    x_range = np.linspace(x.min(), x.max(), 100)
    y_pred = slope * x_range + intercept
    return {
        'scatter_points': filtered[['aht', 'csat']].to_dict('records'),
        'regression_line': {
            'x': x_range.tolist(), 'y': y_pred.tolist(),
            'slope': slope, 'intercept': intercept, 'r_squared': r ** 2
        }
    }

@app.get("/api/filtered-data")
def get_filtered_data(region: str = None, team: str = None,
                      start_date: str = None, end_date: str = None,
                      min_calls: int = None, max_calls: int = None):
    filtered = _filter(df, region, team, start_date, end_date, min_calls, max_calls)
    return filtered.to_dict('records')

@app.get("/api/realtime-metrics")
def get_realtime_metrics():
    latest = df[df['date'] == df['date'].max()]
    return {
        'timestamp': datetime.now().isoformat(),
        'total_calls': int(latest['total_calls'].sum()),
        'avg_csat': round(np.nanmean(latest['csat']), 2),
        'avg_aht': round(np.nanmean(latest['aht']), 2),
        'avg_fcr': round(np.nanmean(latest['fcr']), 2),
        'avg_churn': round(np.nanmean(latest['churn']) * 100, 2),
        'avg_occupancy': round(np.nanmean(latest['occupancy']) * 100, 2),
        'regional_breakdown': latest.groupby('region').agg({
            'total_calls': 'sum', 'csat': lambda x: round(np.nanmean(x), 2),
            'aht': lambda x: round(np.nanmean(x), 2)
        }).to_dict('index')
    }

@app.get("/api/export-data")
def export_data(format: str = "json", region: str = None, team: str = None,
                start_date: str = None, end_date: str = None):
    filtered = _filter(df, region, team, start_date, end_date)
    export_df = filtered.copy()
    export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')

    if format == "csv":
        return {"data": export_df.to_csv(index=False),
                "filename": "customer_support_data.csv",
                "content_type": "text/csv"}
    if format == "excel":
        try:
            import io, base64
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, sheet_name='Customer Support Data', index=False)
            buffer.seek(0)
            return {
                "data": base64.b64encode(buffer.getvalue()).decode('utf-8'),
                "filename": "customer_support_data.xlsx",
                "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "is_binary": True
            }
        except Exception:
            return {"data": export_df.to_json(orient='records'),
                    "filename": "customer_support_data.json",
                    "content_type": "application/json"}
    return export_df.to_dict('records')

# ---------- RUN SERVER ----------
# Vercel expects the app to be named 'app'
app = app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
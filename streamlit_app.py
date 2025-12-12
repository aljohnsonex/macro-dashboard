"""Macro Economic Data Dashboard - Consumer Health visualization"""

import streamlit as st
import pandas as pd
import pandas_gbq
import plotly.express as px
import plotly.graph_objects as go
from google.oauth2 import service_account
import json
import os
from dotenv import load_dotenv
from datetime import timedelta
import math

# Constants
PERIOD_DAYS = {'5yr': 1825, '10yr': 3650, '15yr': 5475}
DEFAULT_PERIOD = '5yr'
STEP_COLORS = {
    'red': 'rgba(255,0,0,0.8)',
    'green': 'rgba(0,128,0,0.8)',
    'grey': 'rgba(128,128,128,0.8)'
}
CHART_CONFIG = {
    'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
        'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian',
        'toggleSpikelines', 'toImage'],
    'displayModeBar': False, 'displaylogo': False, 'doubleClick': False, 'doubleClickDelay': 0
}

load_dotenv()
st.set_page_config(page_title="Macro Economic Dashboard", layout="wide")

def get_credentials():
    """Load GCP credentials from GCP_CREDENTIALS_JSON or GCP_CREDENTIALS_FILE."""
    creds_file = os.getenv("GCP_CREDENTIALS_FILE")
    if creds_file and os.path.exists(creds_file):
        try:
            with open(creds_file, 'r') as f:
                return service_account.Credentials.from_service_account_info(
                    json.load(f), scopes=["https://www.googleapis.com/auth/bigquery"]
                )
        except Exception as e:
            st.error(f"Error reading credentials file: {e}")
            st.stop()
    
    creds_json = os.getenv("GCP_CREDENTIALS_JSON")
    if not creds_json:
        st.error("GCP_CREDENTIALS_FILE or GCP_CREDENTIALS_JSON must be set in .env file")
        st.stop()
    
    creds_json = creds_json.strip().strip('"').strip("'").replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
    try:
        return service_account.Credentials.from_service_account_info(
            json.loads(creds_json), scopes=["https://www.googleapis.com/auth/bigquery"]
        )
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in GCP_CREDENTIALS_JSON: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        st.stop()

def get_column_value(df, col_name, default=None):
    """Safely extract column value from DataFrame."""
    if col_name in df.columns and len(df) > 0 and pd.notna(df[col_name].iloc[0]):
        return df[col_name].iloc[0]
    return default

def determine_date_format(df):
    """Determine date format based on data granularity."""
    if len(df) < 2:
        return '%B %d, %Y'
    df_sorted = df.sort_values('date')
    date_diffs = df_sorted['date'].diff().dropna()
    if len(date_diffs) == 0:
        return '%B %d, %Y'
    median_diff = date_diffs.median()
    if 25 <= median_diff.days <= 35 or 85 <= median_diff.days <= 95:
        return '%B %Y'
    return '%B %d, %Y'

def format_percentile(p):
    """Format percentile with ordinal suffix."""
    p_int = int(round(p))
    if 10 <= p_int % 100 <= 20:
        return f"{p_int}th"
    suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(p_int % 10, 'th')
    return f"{p_int}{suffix}"

def calculate_percentiles(df, value_col, is_growth, needs_conversion):
    """Calculate 20th and 80th percentiles from DataFrame."""
    df_valid = df.dropna(subset=[value_col])
    if len(df_valid) == 0:
        return None, None
    p20_raw = df_valid[value_col].quantile(0.20)
    p80_raw = df_valid[value_col].quantile(0.80)
    for attr in ['item']:
        if hasattr(p20_raw, attr):
            p20_raw = getattr(p20_raw, attr)()
        if hasattr(p80_raw, attr):
            p80_raw = getattr(p80_raw, attr)()
    if is_growth and not needs_conversion:
        return float(p20_raw) / 100, float(p80_raw) / 100
    return float(p20_raw), float(p80_raw)

def get_percentile_colors(percentile_desired):
    """Get colors for 20th and 80th percentile lines based on desired percentile."""
    percentile_lower = str(percentile_desired).lower() if percentile_desired else ''
    color_20th = 'red' if percentile_lower == 'high' else 'green' if percentile_lower == 'low' else 'grey'
    color_80th = 'green' if percentile_lower == 'high' else 'red' if percentile_lower == 'low' else 'grey'
    return color_20th, color_80th

def calculate_y_range(valid_data, percentile_20, percentile_80):
    """Calculate y-axis range including data and percentiles."""
    if len(valid_data) == 0:
        return None
    data_min = valid_data.min().item() if hasattr(valid_data.min(), 'item') else valid_data.min()
    data_max = valid_data.max().item() if hasattr(valid_data.max(), 'item') else valid_data.max()
    if pd.notna(data_min) and pd.notna(data_max):
        y_min = min(float(data_min), percentile_20 if percentile_20 is not None else float('inf'), 0.0)
        y_max = max(float(data_max), percentile_80 if percentile_80 is not None else float('-inf'), 0.0)
        y_padding = (y_max - y_min) * 0.1
        y_range_min, y_range_max = y_min - y_padding, y_max + y_padding
        if pd.notna(y_range_min) and pd.notna(y_range_max) and not (math.isinf(y_range_min) or math.isinf(y_range_max)):
            return [y_range_min, y_range_max]
    return None

def create_line_chart(df_plot, value_col, metric_name, tooltip_date_format, yaxis_tickformat, 
                     y_range, percentile_period_display, format_value, color_20th, color_80th,
                     df_percentile_valid, df_percentile, is_growth, needs_conversion):
    """Create and configure the line chart."""
    fig = px.line(df_plot, x='date', y=value_col, labels={value_col: metric_name or 'Value'}, 
                  custom_data=['percentile_rank'])
    
    fig.update_traces(
        hovertemplate=f'<b><u>{metric_name or "Value"}</u></b> - %{{x|{tooltip_date_format}}}<br>' +
                     f'<b>Value:</b> %{{y:{yaxis_tickformat}}} (%{{customdata[0]:.0f}}th percentile)<br>' +
                     f'<i>Percentiles are calculated based on {percentile_period_display} trailing period from most recent date</i>' +
                     '<extra></extra>',
        line_color='black'
    )
    
    annotations = []
    df_plot_valid = df_plot.dropna(subset=[value_col]) if len(df_plot) > 0 else pd.DataFrame()
    if len(df_plot_valid) > 0:
        most_recent = df_plot_valid.iloc[-1]
        val = most_recent[value_col]
        if pd.notna(val):
            annotations.append(dict(
                x=most_recent['date'], y=val, text=format_value(val), showarrow=False,
                xanchor='center', yanchor='bottom', font=dict(size=8, color='black', family='Arial'),
                bgcolor='white', bordercolor='black', borderwidth=1, yshift=10
            ))
    
    fig.update_layout(
        height=250, width=800, xaxis_title="", yaxis_title="",
        hovermode='closest', dragmode=False, plot_bgcolor='white', paper_bgcolor='white',
        hoverlabel=dict(align='left', bgcolor='white', bordercolor='black', font_size=12, 
                       font_family='Arial', font_color='black'),
        font=dict(family='Arial'),
        margin=dict(l=50, r=20, t=0, b=0),
        xaxis=dict(showgrid=False, showline=False, zeroline=False, fixedrange=True,
            tickformat='%b %Y', linecolor='black', tickfont=dict(color='black', size=12, family='Arial'),
            showticklabels=True, tickangle=0),
        yaxis=dict(showgrid=False, showline=True, zeroline=False,
            range=y_range if (y_range and isinstance(y_range, list) and len(y_range) == 2) else None,
            fixedrange=False, rangemode='tozero', tickformat=yaxis_tickformat,
            linecolor='black', tickfont=dict(color='black', family='Arial'))
    )
    
    fig.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.5, line_width=1)
    
    if len(df_percentile_valid) > 0:
        percentile_20, percentile_80 = calculate_percentiles(df_percentile, value_col, is_growth, needs_conversion)
        min_date_plot = df_plot_valid['date'].min() if len(df_plot_valid) > 0 and 'date' in df_plot_valid.columns else None
        
        if pd.notna(percentile_20) and min_date_plot is not None:
            fig.add_hline(y=percentile_20, line_dash="dot", line_color=color_20th, line_width=1)
            annotations.append(dict(x=min_date_plot, y=percentile_20, text="20th Percentile", showarrow=False,
                xanchor='left', yanchor='top', font=dict(size=9, color=color_20th, family='Arial'),
                bgcolor='white', bordercolor=color_20th, borderwidth=1, xshift=5))
        if pd.notna(percentile_80) and min_date_plot is not None:
            fig.add_hline(y=percentile_80, line_dash="dot", line_color=color_80th, line_width=1)
            annotations.append(dict(x=min_date_plot, y=percentile_80, text="80th Percentile", showarrow=False,
                xanchor='left', yanchor='bottom', font=dict(size=9, color=color_80th, family='Arial'),
                bgcolor='white', bordercolor=color_80th, borderwidth=1, xshift=5))
    
    if annotations:
        fig.update_layout(annotations=annotations)
    
    return fig, df_plot_valid

def create_gauge_chart(most_recent_percentile, current_value_text, color_20th, color_80th, 
                      percentile_period_display):
    """Create and configure the gauge chart."""
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge",
        value=most_recent_percentile,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': current_value_text, 'font': {'size': 12, 'family': 'Arial'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "black",
            'steps': [
                {'range': [0, 20], 'color': STEP_COLORS.get(color_20th, STEP_COLORS['grey'])},
                {'range': [80, 100], 'color': STEP_COLORS.get(color_80th, STEP_COLORS['grey'])}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 1,
                'value': most_recent_percentile
            }
        }
    ))
    
    percentile_label_color = color_20th if most_recent_percentile <= 20 else (color_80th if most_recent_percentile >= 80 else 'black')
    gauge_fig.add_annotation(
        x=0.5, y=0.20,
        text=f"<b>{format_percentile(most_recent_percentile)}</b> percentile<br><i>vs. past {percentile_period_display}s</i>",
        showarrow=False, font=dict(size=12, family='Arial', color=percentile_label_color),
        bgcolor='white', xref="paper", yref="paper"
    )
    gauge_fig.update_layout(
        height=210, width=210, font={'family': 'Arial', 'color': 'black'},
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=40, r=40, t=0, b=20)
    )
    return gauge_fig

def process_metric(df_metric, percentile_period, start_date):
    """Process a single metric and create its charts."""
    if df_metric.empty or 'date' not in df_metric.columns:
        return None
    
    value_col = 'display_value' if 'display_value' in df_metric.columns else 'value'
    display_format = get_column_value(df_metric, 'display_format')
    is_growth = 'growth' in str(display_format).lower() if display_format else False
    percentile_desired = get_column_value(df_metric, 'percentile_desired')
    metric_name = get_column_value(df_metric, 'metric_name')
    primary_source = get_column_value(df_metric, 'primary_source')
    color_20th, color_80th = get_percentile_colors(percentile_desired)
    
    max_date = df_metric['date'].max()
    min_date = df_metric['date'].min()
    percentile_start_date = max(max_date - timedelta(days=PERIOD_DAYS.get(percentile_period, 1825)), min_date)
    df_percentile = df_metric[df_metric['date'] >= percentile_start_date].copy()
    df_filtered = df_metric[(df_metric['date'] >= pd.to_datetime(start_date)) & (df_metric['date'] <= max_date)].copy()
    
    needs_conversion = is_growth and len(df_filtered) > 0 and df_filtered[value_col].abs().max() < 1.0
    df_plot = df_filtered.copy()
    if is_growth and not needs_conversion:
        df_plot[value_col] = df_plot[value_col] / 100
    
    def format_value(val):
        return f"{(val * 100):.2f}%" if is_growth else f"{val:,.2f}"
    
    yaxis_tickformat = '.2%' if is_growth else ',.2f'
    percentile_20_for_range, percentile_80_for_range = calculate_percentiles(df_percentile, value_col, is_growth, needs_conversion)
    
    valid_data = df_plot[value_col].dropna() if len(df_plot) > 0 else pd.Series()
    y_range = calculate_y_range(valid_data, percentile_20_for_range, percentile_80_for_range)
    
    df_percentile_valid = df_percentile.dropna(subset=[value_col]) if len(df_percentile) > 0 else pd.DataFrame()
    df_plot['percentile_rank'] = 0
    if len(df_percentile_valid) > 0:
        percentile_values = df_percentile_valid[value_col].values
        df_plot['percentile_rank'] = df_filtered[value_col].apply(
            lambda val: (percentile_values <= val).sum() / len(percentile_values) * 100 if pd.notna(val) else 0
        ).fillna(0)
    
    tooltip_date_format = determine_date_format(df_plot)
    fig, df_plot_valid = create_line_chart(df_plot, value_col, metric_name, tooltip_date_format, 
                                          yaxis_tickformat, y_range, percentile_period, format_value,
                                          color_20th, color_80th, df_percentile_valid, df_percentile,
                                          is_growth, needs_conversion)
    
    most_recent_percentile = 0
    most_recent_value = None
    if len(df_plot_valid) > 0:
        most_recent = df_plot_valid.iloc[-1]
        most_recent_percentile = most_recent.get('percentile_rank', 0) if 'percentile_rank' in df_plot_valid.columns else 0
        if pd.isna(most_recent_percentile):
            most_recent_percentile = 0
        most_recent_value = most_recent.get(value_col)
    
    current_value_text = ""
    if most_recent_value is not None and pd.notna(most_recent_value):
        current_date = most_recent.get('date')
        if current_date is not None:
            date_format = determine_date_format(df_plot)
            date_str = current_date.strftime(date_format) if isinstance(current_date, pd.Timestamp) else str(current_date)
            current_value_text = f"Current Value: <b>{format_value(most_recent_value)}</b><br><i>as of {date_str}</i>"
        else:
            current_value_text = f"Current Value: <b>{format_value(most_recent_value)}</b>"
    
    gauge_fig = create_gauge_chart(most_recent_percentile, current_value_text, color_20th, 
                                   color_80th, percentile_period)
    
    return {
        'metric_name': metric_name,
        'primary_source': primary_source,
        'fig': fig,
        'gauge_fig': gauge_fig
    }

# Load CSS and JavaScript
st.markdown("""
<style>
    h1 { border-bottom: 4px solid black !important; padding-bottom: 10px !important; }
    * { font-family: Arial, sans-serif !important; }
    .main .block-container, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main, #root {
        background-color: white !important;
        min-width: 1200px !important;
        max-width: 1200px !important;
        width: 1200px !important;
        margin: 0 auto !important;
        box-sizing: border-box !important;
    }
    .main .block-container { padding: 0.5rem 1rem !important; }
    [data-testid="stSidebar"] { position: fixed !important; }
    .element-container { padding-left: 0 !important; padding-right: 0 !important; }
    div:has(div[style*="background-color: black"][style*="color: white"]) + * {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    div[data-testid="column"] { padding: 0.1rem !important; }
    div[data-testid="column"]:nth-of-type(2) { padding-top: 2rem !important; }
    div[data-testid="column"]:nth-of-type(3) { padding-top: 0 !important; margin-top: -1rem !important; }
    
    /* Reduce spacing between metric rows */
    .element-container:has([data-testid="column"]) ~ .element-container:has([data-testid="column"]) {
        margin-top: -20rem !important;
        padding-top: 0 !important;
    }
    .block-container > div > div > .element-container:has([data-testid="column"]) ~ .element-container:has([data-testid="column"]) {
        margin-top: -20rem !important;
        padding-top: 0 !important;
    }
    div:has([data-testid="column"]) ~ div:has([data-testid="column"]) {
        margin-top: -20rem !important;
    }
    .element-container:has([data-testid="column"]) ~ .element-container:has([data-testid="column"]) [data-testid="stPlotlyChart"] {
        margin-top: -20rem !important;
    }
    .element-container:has([data-testid="column"]) ~ .element-container:has([data-testid="stMarkdown"]) {
        margin-top: -8rem !important;
    }
    
    /* Radio button styling */
    div[data-testid="stRadio"] > div { display: flex; gap: 2px; }
    div[data-testid="stRadio"] > div > label {
        background-color: transparent;
        border: 1px solid #ccc;
        border-radius: 3px;
        padding: 2px 8px;
        margin: 0;
        cursor: pointer;
        font-size: 10px !important;
        position: relative;
    }
    div[data-testid="stRadio"] > div > label:has(input[type="radio"]:checked) {
        background-color: white !important;
        border-color: #333;
        font-weight: bold;
    }
    div[data-testid="stRadio"] > div > label > div[data-baseweb="radio"],
    div[data-testid="stRadio"] input[type="radio"],
    div[data-testid="stRadio"] > div > label > span[data-baseweb="radio"],
    div[data-testid="stRadio"] > div > label::before {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Date input styling */
    div[data-testid="stDateInput"] > div,
    div[data-baseweb="input"] {
        border: 1px solid #ccc !important;
        border-radius: 3px !important;
        background-color: white !important;
        padding: 2px 4px !important;
    }
    div[data-testid="stDateInput"] > div > div,
    div[data-baseweb="input"] > div { background-color: white !important; }
    div[data-testid="stDateInput"] input[type="text"],
    div[data-testid="stDateInput"] input[type="date"] {
        font-family: Arial, sans-serif !important;
        background-color: white !important;
        font-size: 10px !important;
        padding: 2px 4px !important;
        height: 24px !important;
    }
    
    /* Chart cropping */
    div[data-testid="stPlotlyChart"] {
        overflow: hidden !important;
    }
    div[data-testid="stPlotlyChart"] .js-plotly-plot {
        margin-top: 5px !important;
        margin-bottom: -20px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="stPlotlyChart"] svg {
        overflow: visible !important;
    }
    div[data-testid="column"]:nth-of-type(3) div[data-testid="stPlotlyChart"] .js-plotly-plot {
        margin-bottom: -20px !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="column"]:nth-of-type(2) div[data-testid="stPlotlyChart"] .js-plotly-plot {
        margin-top: 0px !important;
        padding-top: 0 !important;
    }
</style>
<script>
    function enforceFixedWidth() {
        ['.main .block-container', '.stApp', '[data-testid="stAppViewContainer"]', 
         '[data-testid="stHeader"]', '.main', '#root'].forEach(function(selector) {
            document.querySelectorAll(selector).forEach(function(el) {
                if (el) {
                    ['min-width', 'max-width', 'width', 'margin', 'box-sizing'].forEach(function(prop) {
                        el.style.setProperty(prop, prop === 'box-sizing' ? 'border-box' : 
                            prop === 'margin' ? '0 auto' : '1200px', 'important');
                    });
                }
            });
        });
    }
    
    function formatDateInput() {
        document.querySelectorAll('div[data-testid="stDateInput"] input[type="text"]').forEach(function(input) {
            function formatDate() {
                if (input.value) {
                    var date = new Date(input.value);
                    if (!isNaN(date.getTime())) {
                        input.value = String(date.getMonth() + 1).padStart(2, '0') + '/' + 
                                      String(date.getDate()).padStart(2, '0') + '/' + date.getFullYear();
                    }
                }
            }
            formatDate();
            input.addEventListener('change', formatDate);
            input.addEventListener('blur', formatDate);
        });
    }
    
    function changeRadioColor() {
        document.querySelectorAll('div[data-testid="stRadio"] div[data-baseweb="radio-mark"]').forEach(function(mark) {
            mark.style.backgroundColor = 'black';
            mark.style.borderColor = 'black';
        });
        document.querySelectorAll('div[data-testid="stRadio"] input[type="radio"]:checked').forEach(function(radio) {
            radio.style.accentColor = 'black';
            var mark = radio.closest('label')?.querySelector('div[data-baseweb="radio-mark"]');
            if (mark) {
                mark.style.backgroundColor = 'black';
                mark.style.borderColor = 'black';
            }
        });
    }
    
    function reduceRowSpacing() {
        var columnContainers = document.querySelectorAll('[data-testid="column"]');
        var firstRowFound = false;
        columnContainers.forEach(function(col) {
            var parent = col.closest('.element-container');
            if (!parent) return;
            
            var rowIndex = Array.from(parent.parentElement.children).indexOf(parent);
            if (rowIndex === 0) {
                firstRowFound = true;
                return;
            }
            
            if (firstRowFound && rowIndex > 0) {
                parent.style.marginTop = '-20rem';
                parent.style.paddingTop = '0';
                parent.style.marginBottom = '0';
                parent.style.paddingBottom = '0';
                
                var plotlyCharts = parent.querySelectorAll('[data-testid="stPlotlyChart"]');
                plotlyCharts.forEach(function(chart) {
                    chart.style.marginTop = '-15rem';
                    chart.style.marginBottom = '0';
                });
            }
        });
    }
    
    function applyRowSpacing() {
        reduceRowSpacing();
        setTimeout(reduceRowSpacing, 500);
        setTimeout(reduceRowSpacing, 1000);
        setTimeout(reduceRowSpacing, 2000);
    }
    
    // Initialize functions
    enforceFixedWidth();
    [formatDateInput, changeRadioColor].forEach(function(fn) {
        window.addEventListener('load', function() { setTimeout(fn, 500); });
        new MutationObserver(function() { setTimeout(fn, 100); }).observe(document.body, { childList: true, subtree: true });
    });
    
    window.addEventListener('load', enforceFixedWidth);
    document.addEventListener('DOMContentLoaded', enforceFixedWidth);
    var observer = new MutationObserver(enforceFixedWidth);
    if (document.body) observer.observe(document.body, { childList: true, subtree: true, attributes: true });
    if (document.documentElement) observer.observe(document.documentElement, { childList: true, subtree: true, attributes: true });
    
    window.addEventListener('load', applyRowSpacing);
    document.addEventListener('DOMContentLoaded', applyRowSpacing);
    var spacingObserver = new MutationObserver(function() {
        setTimeout(reduceRowSpacing, 100);
    });
    if (document.body) spacingObserver.observe(document.body, { childList: true, subtree: true });
</script>
""", unsafe_allow_html=True)

# Main application
st.title("Macro Data Overview")

project_id = os.getenv("BQ_PROJECT_ID")
dataset_id = os.getenv("BQ_DATASET_ID")
table_id = "macro_consolidated"

if not project_id or not dataset_id:
    st.error("BQ_PROJECT_ID and BQ_DATASET_ID must be set in .env file")
    st.stop()

credentials = get_credentials()

@st.cache_data(ttl=3600)
def load_data():
    """Load all Consumer Health series from macro_consolidated table."""
    try:
        query = f"""
        SELECT date, value, display_value, metric_name, units, category, display_format, percentile_desired, primary_source
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE category = 'Consumer Health'
        ORDER BY metric_name, date ASC
        """
        df = pandas_gbq.read_gbq(query, project_id=project_id, credentials=credentials, use_bqstorage_api=True)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

with st.spinner("Loading Consumer Health data from BigQuery..."):
    df = load_data()

if df.empty:
    st.warning("No Consumer Health data found.")
else:
    category = get_column_value(df, 'category')
    
    if 'date' in df.columns:
        max_date = df['date'].max()
        min_date = df['date'].min()
        
        if 'percentile_period' not in st.session_state:
            st.session_state.percentile_period = DEFAULT_PERIOD
        
        def to_date(d):
            return d.date() if isinstance(d, pd.Timestamp) else d
        
        default_min_date = max(to_date(max_date - timedelta(days=PERIOD_DAYS[DEFAULT_PERIOD])), to_date(min_date))
        min_date_input, max_date_input = to_date(min_date), to_date(max_date)
        
        col1, col2, col3 = st.columns([0.5, 2, 1.5])
        with col3:
            col_title1, col_options1 = st.columns([0.8, 2.2], gap="small")
            with col_title1:
                st.markdown("<div style='display: flex; align-items: center; justify-content: flex-start; padding-top: 10px; min-height: 10px; padding-right: 4px; white-space: nowrap;'><p style='margin: 0; font-size: 12px;'><b>Percentile Period</b></p></div>", unsafe_allow_html=True)
            with col_options1:
                options = list(PERIOD_DAYS.keys())
                idx = options.index(st.session_state.percentile_period) if st.session_state.percentile_period in options else 0
                percentile_period = st.radio("", options=options, index=idx, key="percentile_period_filter", horizontal=True, label_visibility="collapsed")
            st.session_state.percentile_period = percentile_period
            
            col_title2, col_input2 = st.columns([0.8, 2.2], gap="small")
            with col_title2:
                st.markdown("<div style='display: flex; align-items: center; justify-content: flex-start; min-height: 38px; padding-top: 6px; padding-right: 0px;'><p style='margin: 0; font-size: 12px;'><b>Start Date</b></p></div>", unsafe_allow_html=True)
            with col_input2:
                start_date = st.date_input("", value=default_min_date, min_value=min_date_input, max_value=max_date_input, key="start_date_filter", label_visibility="collapsed")
    else:
        percentile_period = st.session_state.get('percentile_period', DEFAULT_PERIOD)
        start_date = None
    
    if category:
        st.markdown(f"<div style='background-color: black; color: white; padding-left: 5px; width: 100%; margin-bottom: 0.25rem;'><p style='font-size: 20px; font-family: Arial, sans-serif; margin: 0; text-align: left;'><b>{category}</b></p></div>", unsafe_allow_html=True)
    
    unique_metrics = df['metric_name'].unique() if 'metric_name' in df.columns else []
    
    for metric_name in sorted(unique_metrics):
        df_metric = df[df['metric_name'] == metric_name].copy()
        result = process_metric(df_metric, percentile_period, start_date)
        
        if result:
            col1, col2, col3 = st.columns([1, 1, 2.5])
            with col1:
                if result['metric_name']:
                    source_text = f"<p style='font-size: 14px; font-family: Arial, sans-serif; margin-top: 0rem; margin-bottom: 0;'>Source: {result['primary_source']}</p>" if result['primary_source'] else ""
                    st.markdown(f"<div style='text-align: center; padding-top: 3.5rem;'><p style='font-size: 18px; font-family: Arial, sans-serif; margin: 0;'><b>{result['metric_name']}</b></p>{source_text}</div>", unsafe_allow_html=True)
            with col2:
                st.markdown("<div style='padding-top: 1rem;'></div>", unsafe_allow_html=True)
                st.plotly_chart(result['gauge_fig'], use_container_width=False, config=CHART_CONFIG)
            with col3:
                st.plotly_chart(result['fig'], use_container_width=False, config=CHART_CONFIG)

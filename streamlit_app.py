"""Macro Economic Data Dashboard"""

import streamlit as st
import pandas as pd
import pandas_gbq
import plotly.graph_objects as go
from google.oauth2 import service_account
import json
import os
from dotenv import load_dotenv
from datetime import timedelta
import math

# Constants
PERIOD_DAYS = {'5yr': 1825, '10yr': 3650, '15yr': 5475}
DEFAULT_PERIOD = '15yr'
CATEGORY_ORDER = [
    'Consumer Health', 'Housing & Construction', 'Production & Surveys',
    'Labor', 'Inflation & Money', 'Government & Markets'
]
HIDDEN_SERIES = ['Federal TTM Budget (Deficit) as Percent of GDP']
CHART_CONFIG = {
    'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d',
        'autoScale2d', 'resetScale2d', 'hoverClosestCartesian', 'hoverCompareCartesian',
        'toggleSpikelines', 'toImage'],
    'displayModeBar': False, 'displaylogo': False, 'doubleClick': False, 'doubleClickDelay': 0
}

load_dotenv()
st.set_page_config(page_title="Macro Economic Dashboard", layout="wide")

def get_credentials():
    """Load GCP credentials from Streamlit secrets or environment variables."""
    # First check Streamlit secrets for "gcp" section
    try:
        if hasattr(st, 'secrets') and 'gcp' in st.secrets:
            try:
                return service_account.Credentials.from_service_account_info(
                    st.secrets["gcp"], scopes=["https://www.googleapis.com/auth/bigquery"]
                )
            except Exception as e:
                st.error(f"Error loading credentials from st.secrets['gcp']: {e}")
                st.stop()
        
        # Fallback: check for GCP_CREDENTIALS_FILE as dict (TOML section)
        if hasattr(st, 'secrets') and 'GCP_CREDENTIALS_FILE' in st.secrets:
            creds_data = st.secrets['GCP_CREDENTIALS_FILE']
            if isinstance(creds_data, dict):
                try:
                    return service_account.Credentials.from_service_account_info(
                        creds_data, scopes=["https://www.googleapis.com/auth/bigquery"]
                    )
                except Exception as e:
                    st.error(f"Error loading credentials from secrets dict: {e}")
                    st.stop()
            elif isinstance(creds_data, str) and os.path.exists(creds_data):
                try:
                    with open(creds_data, 'r') as f:
                        return service_account.Credentials.from_service_account_info(
                            json.load(f), scopes=["https://www.googleapis.com/auth/bigquery"]
                        )
                except Exception as e:
                    st.error(f"Error reading credentials file from secrets: {e}")
                    st.stop()
        
        # Fallback: check for GCP_CREDENTIALS_JSON
        if hasattr(st, 'secrets') and 'GCP_CREDENTIALS_JSON' in st.secrets:
            creds_json = st.secrets['GCP_CREDENTIALS_JSON']
            if creds_json:
                if isinstance(creds_json, dict):
                    try:
                        return service_account.Credentials.from_service_account_info(
                            creds_json, scopes=["https://www.googleapis.com/auth/bigquery"]
                        )
                    except Exception as e:
                        st.error(f"Error loading credentials from secrets JSON dict: {e}")
                        st.stop()
                else:
                    creds_json = str(creds_json).strip().strip('"').strip("'").replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
                    try:
                        return service_account.Credentials.from_service_account_info(
                            json.loads(creds_json), scopes=["https://www.googleapis.com/auth/bigquery"]
                        )
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON in GCP_CREDENTIALS_JSON from secrets: {e}")
                        st.stop()
                    except Exception as e:
                        st.error(f"Error loading credentials from secrets: {e}")
                        st.stop()
    except Exception as e:
        # If secrets access fails, fall through to environment variables
        pass
    
    # Fall back to environment variables
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
        st.error("GCP credentials must be set in Streamlit secrets (st.secrets['gcp']) or .env file (GCP_CREDENTIALS_FILE/GCP_CREDENTIALS_JSON)")
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

def calculate_percentiles(df, value_col):
    """Calculate 20th and 80th percentiles from DataFrame."""
    df_valid = df.dropna(subset=[value_col])
    if len(df_valid) == 0:
        return None, None
    p20 = float(df_valid[value_col].quantile(0.20))
    p80 = float(df_valid[value_col].quantile(0.80))
    return p20, p80

def get_percentile_colors(percentile_desired):
    """Get colors for 20th and 80th percentile lines based on desired percentile."""
    percentile_lower = str(percentile_desired).lower() if percentile_desired else ''
    is_high = percentile_lower == 'high'
    color_20th = 'red' if is_high else 'green' if percentile_lower == 'low' else 'grey'
    color_80th = 'green' if is_high else 'red' if percentile_lower == 'low' else 'grey'
    return color_20th, color_80th

def calculate_y_range(valid_data, percentile_20, percentile_80):
    """Calculate y-axis range including data and percentiles."""
    if len(valid_data) == 0:
        return None
    data_min = float(valid_data.min())
    data_max = float(valid_data.max())
    if pd.notna(data_min) and pd.notna(data_max):
        y_min = min(data_min, percentile_20 if percentile_20 is not None else float('inf'), 0.0)
        y_max = max(data_max, percentile_80 if percentile_80 is not None else float('-inf'), 0.0)
        y_padding = (y_max - y_min) * 0.1
        y_range_min, y_range_max = y_min - y_padding, y_max + y_padding
        if pd.notna(y_range_min) and pd.notna(y_range_max) and not (math.isinf(y_range_min) or math.isinf(y_range_max)):
            return [y_range_min, y_range_max]
    return None

def normalize_category_for_sorting(cat_name):
    """Normalize category name for consistent sorting."""
    if not cat_name or pd.isna(cat_name):
        return None
    cat_lower = str(cat_name).strip().lower()
    if 'consumer health' in cat_lower:
        return 'Consumer Health'
    elif 'housing' in cat_lower and ('construction' in cat_lower or 'building' in cat_lower):
        return 'Housing & Construction'
    elif 'production' in cat_lower and 'survey' in cat_lower:
        return 'Production & Surveys'
    elif 'labor' in cat_lower:
        return 'Labor'
    elif 'inflation' in cat_lower and 'money' in cat_lower:
        return 'Inflation & Money'
    elif 'government' in cat_lower and 'market' in cat_lower:
        return 'Government & Markets'
    return cat_name

def sort_categories(categories):
    """Sort categories according to CATEGORY_ORDER."""
    category_mapping = {}
    for cat in categories:
        if not cat or pd.isna(cat):
            continue
        normalized = normalize_category_for_sorting(cat)
        for idx, ordered_cat in enumerate(CATEGORY_ORDER):
            if normalized and normalized.lower() == ordered_cat.lower():
                category_mapping[cat] = idx
                break
        if cat not in category_mapping:
            category_mapping[cat] = 9999
    return sorted([c for c in categories if c and not pd.isna(c)], 
                  key=lambda cat: category_mapping.get(cat, 9999))

def filter_metrics(metrics):
    """Filter out hidden series."""
    return [m for m in metrics if str(m).strip() not in HIDDEN_SERIES]

def prepare_metric_data(df_metric, percentile_period, start_date):
    """Extract and prepare common metric data."""
    if df_metric.empty or 'date' not in df_metric.columns:
        return None
    
    value_col = 'display_value' if 'display_value' in df_metric.columns else 'value'
    display_format = get_column_value(df_metric, 'display_format')
    metric_name = get_column_value(df_metric, 'metric_name')
    metric_name_lower = str(metric_name).lower() if metric_name else ''
    is_percentage_name = any(term in metric_name_lower for term in ['rate', 'spread', 'percent'])
    is_growth = 'growth' in str(display_format).lower() if display_format else False
    is_growth = is_growth or is_percentage_name
    
    max_date = df_metric['date'].max()
    min_date = df_metric['date'].min()
    percentile_start_date = max(max_date - timedelta(days=PERIOD_DAYS.get(percentile_period, 1825)), min_date)
    df_percentile = df_metric[df_metric['date'] >= percentile_start_date].copy()
    
    expected_days = PERIOD_DAYS.get(percentile_period, 1825)
    actual_days = (max_date - percentile_start_date).days if len(df_percentile) > 0 else 0
    has_full_period = actual_days >= (expected_days * 0.95)
    
    if start_date is not None:
        df_filtered = df_metric[(df_metric['date'] >= pd.to_datetime(start_date)) & (df_metric['date'] <= max_date)].copy()
    else:
        df_filtered = df_metric[df_metric['date'] <= max_date].copy()
    
    needs_conversion = is_growth and len(df_filtered) > 0 and df_filtered[value_col].abs().max() < 1.0
    df_plot = df_filtered.copy()
    if is_growth and not needs_conversion:
        df_plot[value_col] = df_plot[value_col] / 100
        df_percentile[value_col] = df_percentile[value_col] / 100
    
    def format_value(val):
        return f"{(val * 100):.2f}%" if is_growth else f"{val:,.2f}"
    
    df_percentile_valid = df_percentile.dropna(subset=[value_col]) if len(df_percentile) > 0 else pd.DataFrame()
    df_plot_valid = df_plot.dropna(subset=[value_col]) if len(df_plot) > 0 else pd.DataFrame()
    
    return {
        'value_col': value_col, 'is_growth': is_growth, 'needs_conversion': needs_conversion,
        'metric_name': metric_name, 'primary_source': get_column_value(df_metric, 'primary_source'),
        'target_min': get_column_value(df_metric, 'target_min'),
        'target_max': get_column_value(df_metric, 'target_max'),
        'color_20th': get_percentile_colors(get_column_value(df_metric, 'percentile_desired'))[0],
        'color_80th': get_percentile_colors(get_column_value(df_metric, 'percentile_desired'))[1],
        'df_percentile': df_percentile, 'df_percentile_valid': df_percentile_valid,
        'df_plot': df_plot, 'df_plot_valid': df_plot_valid,
        'format_value': format_value, 'has_full_period': has_full_period
    }

def create_line_chart(df_plot, value_col, metric_name, tooltip_date_format, yaxis_tickformat, 
                     y_range, percentile_period_display, format_value, color_20th, color_80th,
                     df_percentile_valid, df_percentile, target_min=None, target_max=None, show_zero_axis=True):
    """Create and configure the line chart."""
    fig = go.Figure()
    percentile_data = df_plot['percentile_rank'].fillna(0).values if 'percentile_rank' in df_plot.columns else [0] * len(df_plot)
    
    fig.add_trace(go.Scatter(
        x=df_plot['date'], y=df_plot[value_col], mode='lines',
        name=metric_name or 'Value', customdata=percentile_data, line=dict(color='black')
    ))
    
    fig.update_traces(
        hovertemplate=f'<b><u>{metric_name or "Value"}</u></b> - %{{x|{tooltip_date_format}}}<br>' +
                     f'<b>Value:</b> %{{y:{yaxis_tickformat}}} (%{{customdata:.0f}}th percentile)<br>' +
                     f'<i>Percentiles are calculated based on {percentile_period_display} trailing period from most recent date</i>' +
                     '<extra></extra>'
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
        margin=dict(l=50, r=20, t=10, b=0),
        xaxis=dict(showgrid=False, showline=False, zeroline=False, fixedrange=True,
            tickformat='%b %Y', linecolor='black', tickfont=dict(color='black', size=12, family='Arial'),
            showticklabels=True, tickangle=0),
        yaxis=dict(showgrid=False, showline=True, zeroline=False,
            range=y_range if (y_range and isinstance(y_range, list) and len(y_range) == 2) else None,
            fixedrange=False, rangemode='tozero', tickformat=yaxis_tickformat,
            linecolor='black', tickfont=dict(color='black', family='Arial'))
    )
    
    if show_zero_axis:
        fig.add_hline(y=0, line_dash="dot", line_color="grey", opacity=0.5, line_width=1)
    
    min_date_plot = df_plot_valid['date'].min() if len(df_plot_valid) > 0 and 'date' in df_plot_valid.columns else None
    use_targets = (target_min is not None and pd.notna(target_min)) or (target_max is not None and pd.notna(target_max))
    
    if use_targets:
        if target_min is not None and pd.notna(target_min) and min_date_plot is not None:
            fig.add_hline(y=target_min, line_dash="dot", line_color="green", line_width=1)
            annotations.append(dict(x=min_date_plot, y=target_min, text="Target Min", showarrow=False,
                xanchor='left', yanchor='top', font=dict(size=9, color='green', family='Arial'),
                bgcolor='white', bordercolor='green', borderwidth=1, xshift=5))
        if target_max is not None and pd.notna(target_max) and min_date_plot is not None:
            fig.add_hline(y=target_max, line_dash="dot", line_color="green", line_width=1)
            annotations.append(dict(x=min_date_plot, y=target_max, text="Target Max", showarrow=False,
                xanchor='left', yanchor='bottom', font=dict(size=9, color='green', family='Arial'),
                bgcolor='white', bordercolor='green', borderwidth=1, xshift=5))
    elif len(df_percentile_valid) > 0:
        percentile_20, percentile_80 = calculate_percentiles(df_percentile, value_col)
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
                      percentile_period_display, include_annotations=True, title_below=False,
                      target_min=None, target_max=None, current_value=None,
                      data_min=None, data_max=None, format_value=None, has_full_period=True, is_growth=False):
    """Create and configure the gauge chart."""
    domain_y = [0.3, 1] if title_below else [0, 1]
    use_targets = (target_min is not None and pd.notna(target_min)) or (target_max is not None and pd.notna(target_max))
    
    if use_targets and current_value is not None and pd.notna(current_value) and data_min is not None and data_max is not None:
        threshold_color = 'black'
        
        # Convert target values if they're percentages stored as whole numbers (e.g., 5 for 5%)
        # Check if targets are > 1 and data is < 1, indicating targets need conversion
        if is_growth and target_min is not None and pd.notna(target_min) and target_min > 1 and data_max < 1:
            target_min = target_min / 100
        if is_growth and target_max is not None and pd.notna(target_max) and target_max > 1 and data_max < 1:
            target_max = target_max / 100
        
        # Handle cases where only one target is provided
        if target_min is not None and target_max is not None and pd.notna(target_min) and pd.notna(target_max):
            # Both targets provided - center the target range on the gauge
            target_center = (target_min + target_max) / 2
            target_range_span = abs(target_max - target_min)
            # Make gauge range 2.5x the target range to center it nicely
            gauge_range_span = target_range_span * 2.5
            axis_min = target_center - gauge_range_span / 2
            axis_max = target_center + gauge_range_span / 2
        elif target_min is not None and pd.notna(target_min):
            # Only min target - center around target_min
            target_range_span = abs(data_max - data_min) * 0.3 if data_max is not None and data_min is not None else abs(target_min) * 0.3
            gauge_range_span = target_range_span * 2.5
            axis_min = target_min - gauge_range_span / 2
            axis_max = target_min + gauge_range_span / 2
        elif target_max is not None and pd.notna(target_max):
            # Only max target - center around target_max
            target_range_span = abs(data_max - data_min) * 0.3 if data_max is not None and data_min is not None else abs(target_max) * 0.3
            gauge_range_span = target_range_span * 2.5
            axis_min = target_max - gauge_range_span / 2
            axis_max = target_max + gauge_range_span / 2
        else:
            # Fallback to data range
            axis_min = data_min
            axis_max = data_max
        
        # Final safety checks
        if not pd.notna(axis_min) or math.isinf(axis_min):
            axis_min = data_min - (data_max - data_min) * 0.1 if data_min is not None and data_max is not None else 0
        if not pd.notna(axis_max) or math.isinf(axis_max):
            axis_max = data_max + (data_max - data_min) * 0.1 if data_min is not None and data_max is not None else 100
        
        steps = []
        if target_min is not None and target_max is not None:
            steps.append({'range': [target_min, target_max], 'color': 'rgba(0, 128, 0, 0.3)'})
            if target_min > axis_min:
                steps.append({'range': [axis_min, target_min], 'color': 'rgba(255, 0, 0, 0.3)'})
            if target_max < axis_max:
                steps.append({'range': [target_max, axis_max], 'color': 'rgba(255, 0, 0, 0.3)'})
        elif target_min is not None:
            steps.append({'range': [target_min, axis_max], 'color': 'rgba(0, 128, 0, 0.3)'})
            if target_min > axis_min:
                steps.append({'range': [axis_min, target_min], 'color': 'rgba(255, 0, 0, 0.3)'})
        elif target_max is not None:
            steps.append({'range': [axis_min, target_max], 'color': 'rgba(0, 128, 0, 0.3)'})
            if target_max < axis_max:
                steps.append({'range': [target_max, axis_max], 'color': 'rgba(255, 0, 0, 0.3)'})
        
        gauge_value = float(current_value)
        axis_range = [axis_min, axis_max]
        tickvals, ticktext = [], []
        if target_min is not None and pd.notna(target_min) and axis_min < target_min < axis_max:
            tickvals.append(target_min)
            ticktext.append(format_value(target_min))
        tickvals.append((axis_min + axis_max) / 2)
        ticktext.append('')
        if target_max is not None and pd.notna(target_max) and axis_min < target_max < axis_max:
            tickvals.append(target_max)
            ticktext.append(format_value(target_max))
    else:
        threshold_color = 'black'
        color_20th_rgba = 'rgba(0, 128, 0, 0.3)' if color_20th == 'green' else 'rgba(255, 0, 0, 0.3)'
        color_80th_rgba = 'rgba(0, 128, 0, 0.3)' if color_80th == 'green' else 'rgba(255, 0, 0, 0.3)'
        steps = [
            {'range': [0, 20], 'color': color_20th_rgba},
            {'range': [80, 100], 'color': color_80th_rgba}
        ]
        gauge_value = most_recent_percentile
        axis_range = [None, 100]
        tickvals, ticktext = [0, 50, 100], ['0', '', '100']
    
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge",
        value=gauge_value,
        domain={'x': [0, 1], 'y': domain_y},
        title={'text': '' if title_below else current_value_text, 'font': {'size': 12, 'family': 'Arial'}},
        gauge={
            'axis': {
                'range': axis_range, 'tickwidth': 1, 'tickcolor': "black",
                'tickmode': 'array', 'tickvals': tickvals, 'ticktext': ticktext,
                'tickfont': {'color': 'black', 'family': 'Arial'}
            },
            'bar': {'color': "rgba(0,0,0,0)"}, 'bgcolor': "white",
            'borderwidth': 2, 'bordercolor': "black", 'steps': steps,
            'threshold': {'line': {'color': 'black', 'width': 3}, 'thickness': 1, 'value': gauge_value}
        }
    ))
    
    if include_annotations and not use_targets:
        percentile_label_color = color_20th if most_recent_percentile <= 20 else (color_80th if most_recent_percentile >= 80 else 'black')
        gauge_fig.add_annotation(
            x=0.5, y=0.20,
            text=f"<b>{format_percentile(most_recent_percentile)}</b> percentile<br><i>vs. past {percentile_period_display}s</i>",
            showarrow=False, font=dict(size=12, family='Arial', color=percentile_label_color),
            bgcolor='white', xref="paper", yref="paper"
        )
    
    if title_below and current_value_text:
        if use_targets:
            full_text = current_value_text
        else:
            asterisk = "*" if not has_full_period else ""
            percentile_text = f"<b>{format_percentile(most_recent_percentile)}</b> pct. vs. 15Y{asterisk}"
            full_text = f"{current_value_text}<br><i>{percentile_text}</i>"
        gauge_fig.add_annotation(
            x=0.5, y=.50, text=full_text,
            showarrow=False, font=dict(size=10, family='Arial', color='black'),
            xref="paper", yref="paper", align='center'
        )
        bottom_margin = 40
    else:
        bottom_margin = 20
    
    chart_height = 150 if title_below else 210
    gauge_fig.update_layout(
        height=chart_height, width=210, font={'family': 'Arial', 'color': 'black'},
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(l=40, r=40, t=0, b=40 if title_below else bottom_margin)
    )
    return gauge_fig

def process_metric_gauge_only(df_metric, percentile_period, start_date):
    """Process a single metric and create only its gauge chart (for top section)."""
    data = prepare_metric_data(df_metric, percentile_period, start_date)
    if not data:
        return None
    
    most_recent_percentile = 0
    most_recent_value = None
    if len(data['df_plot_valid']) > 0:
        most_recent = data['df_plot_valid'].iloc[-1]
        if len(data['df_percentile_valid']) > 0:
            percentile_values = data['df_percentile_valid'][data['value_col']].values
            most_recent_percentile = ((percentile_values <= most_recent[data['value_col']]).sum() / len(percentile_values) * 100) if pd.notna(most_recent[data['value_col']]) else 0
        most_recent_value = most_recent.get(data['value_col'])
    
    current_value_text = ""
    as_of_text = ""
    if most_recent_value is not None and pd.notna(most_recent_value):
        current_date = most_recent.get('date')
        if current_date is not None:
            date_format = determine_date_format(data['df_plot'])
            date_str = current_date.strftime(date_format) if isinstance(current_date, pd.Timestamp) else str(current_date)
            current_value_text = f"Value: <b>{data['format_value'](most_recent_value)}</b>"
            as_of_text = f"<i>as of {date_str}</i>"
    
    data_min = float(data['df_percentile_valid'][data['value_col']].min()) if len(data['df_percentile_valid']) > 0 else None
    data_max = float(data['df_percentile_valid'][data['value_col']].max()) if len(data['df_percentile_valid']) > 0 else None
    
    chart_fig = create_gauge_chart(most_recent_percentile, current_value_text, data['color_20th'], 
                                   data['color_80th'], percentile_period, include_annotations=False, title_below=True,
                                   target_min=data['target_min'], target_max=data['target_max'], 
                                   current_value=most_recent_value, data_min=data_min, data_max=data_max,
                                   format_value=data['format_value'], has_full_period=data.get('has_full_period', True),
                                   is_growth=data['is_growth'])
    
    return {
        'metric_name': data['metric_name'],
        'primary_source': data['primary_source'],
        'gauge_fig': chart_fig,
        'as_of_text': as_of_text
    }

def process_metric(df_metric, percentile_period, start_date):
    """Process a single metric and create its charts."""
    data = prepare_metric_data(df_metric, percentile_period, start_date)
    if not data:
        return None
    
    yaxis_tickformat = '.2%' if data['is_growth'] else ',.2f'
    percentile_20_for_range, percentile_80_for_range = calculate_percentiles(data['df_percentile'], data['value_col'])
    
    valid_data = data['df_plot'][data['value_col']].dropna() if len(data['df_plot']) > 0 else pd.Series()
    use_targets = (data['target_min'] is not None and pd.notna(data['target_min'])) or (data['target_max'] is not None and pd.notna(data['target_max']))
    y_range = calculate_y_range(valid_data, data['target_min'] if use_targets else percentile_20_for_range,
                                data['target_max'] if use_targets else percentile_80_for_range)
    
    if len(data['df_percentile_valid']) > 0:
        percentile_values = data['df_percentile_valid'][data['value_col']].values
        data['df_plot']['percentile_rank'] = data['df_plot'][data['value_col']].apply(
            lambda val: (percentile_values <= val).sum() / len(percentile_values) * 100 if pd.notna(val) else 0
        ).fillna(0)
    else:
        data['df_plot']['percentile_rank'] = 0
    
    tooltip_date_format = determine_date_format(data['df_plot'])
    show_zero_axis = 'labor force participation rate' not in str(data['metric_name']).lower()
    fig, df_plot_valid = create_line_chart(
        data['df_plot'], data['value_col'], data['metric_name'], tooltip_date_format, 
        yaxis_tickformat, y_range, percentile_period, data['format_value'],
        data['color_20th'], data['color_80th'], data['df_percentile_valid'], data['df_percentile'],
        target_min=data['target_min'], target_max=data['target_max'], show_zero_axis=show_zero_axis
    )
    
    most_recent_percentile = 0
    most_recent_value = None
    if len(df_plot_valid) > 0:
        most_recent = df_plot_valid.iloc[-1]
        most_recent_percentile = most_recent.get('percentile_rank', 0) if 'percentile_rank' in df_plot_valid.columns else 0
        if pd.isna(most_recent_percentile):
            most_recent_percentile = 0
        most_recent_value = most_recent.get(data['value_col'])
    
    current_value_text = ""
    if most_recent_value is not None and pd.notna(most_recent_value):
        current_date = most_recent.get('date')
        if current_date is not None:
            date_format = determine_date_format(data['df_plot'])
            date_str = current_date.strftime(date_format) if isinstance(current_date, pd.Timestamp) else str(current_date)
            current_value_text = f"Current Value: <b>{data['format_value'](most_recent_value)}</b><br><i>as of {date_str}</i>"
    
    data_min = float(data['df_percentile_valid'][data['value_col']].min()) if len(data['df_percentile_valid']) > 0 else None
    data_max = float(data['df_percentile_valid'][data['value_col']].max()) if len(data['df_percentile_valid']) > 0 else None
    
    gauge_fig = create_gauge_chart(most_recent_percentile, current_value_text, data['color_20th'], 
                                   data['color_80th'], percentile_period,
                                   target_min=data['target_min'], target_max=data['target_max'], 
                                   current_value=most_recent_value, data_min=data_min, data_max=data_max,
                                   format_value=data['format_value'], has_full_period=data.get('has_full_period', True),
                                   is_growth=data['is_growth'])
    
    return {
        'metric_name': data['metric_name'],
        'primary_source': data['primary_source'],
        'fig': fig,
        'gauge_fig': gauge_fig
    }

# CSS and JavaScript
st.markdown("""
<style>
    h1 { border-bottom: 4px solid black !important; padding-bottom: 10px !important; margin-bottom: 20px !important; }
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
    
    .element-container:has([data-testid="column"]):not(:has([data-testid="stPlotlyChart"] [data-key*="line"])) {
        margin-bottom: -100px !important;
        padding-bottom: 0 !important;
    }
    .block-container .element-container:has([data-testid="stPlotlyChart"]):not(:has([data-key*="line"])) {
        margin-top: -20px !important;
        margin-bottom: -100px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    .element-container:has([data-testid="column"]:nth-of-type(1)):has([data-testid="column"]:nth-of-type(2)):has([data-testid="column"]:nth-of-type(3)),
    .element-container:has([data-testid="stPlotlyChart"][data-key*="line"]),
    .block-container > div > div > .element-container:has([data-testid="column"]:nth-of-type(3)) {
        margin-top: 3rem !important;
        margin-bottom: 3rem !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    button[key^="category_pill_"] {
        padding: 6px 12px !important;
        border-radius: 20px !important;
        background-color: white !important;
        color: black !important;
        font-family: Arial, sans-serif !important;
        font-size: 12px !important;
        font-weight: normal !important;
        cursor: pointer !important;
        margin: 0 4px 0 0 !important;
        border: none !important;
        transition: border 0.2s !important;
    }
    button[key^="category_pill_"][data-selected="true"] {
        border: 2px solid black !important;
        font-weight: bold !important;
    }
    
    button[key^="category_"] {
        font-size: 11px !important;
        font-family: Arial, sans-serif !important;
        font-weight: bold !important;
        padding: 4px 10px !important;
        border-radius: 3px !important;
        margin: 0 !important;
        height: 28px !important;
        min-height: 28px !important;
        max-height: 28px !important;
        border: 1px solid black !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    div[data-testid="stDateInput"] {
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        width: auto !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .element-container:has(div[data-testid="stDateInput"]) {
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stDateInput"] > div,
    div[data-baseweb="input"] {
        border: none !important;
        border-radius: 0 !important;
        background-color: white !important;
        padding: 0 !important;
        height: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        display: flex !important;
        align-items: center !important;
        width: auto !important;
        margin: 0 !important;
    }
    div[data-testid="stDateInput"] > div > div,
    div[data-baseweb="input"] > div { 
        background-color: white !important;
        height: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stDateInput"] input[type="text"],
    div[data-testid="stDateInput"] input[type="date"] {
        font-family: Arial, sans-serif !important;
        background-color: white !important;
        font-size: 12px !important;
        padding: 4px 8px !important;
        height: 32px !important;
        min-height: 32px !important;
        max-height: 32px !important;
        border: 1px solid #ccc !important;
        border-radius: 4px !important;
        width: auto !important;
        margin: 0 !important;
    }
    
    div[data-testid="stPlotlyChart"] {
        overflow: hidden !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    div[data-testid="stPlotlyChart"] .js-plotly-plot {
        margin-top: 5px !important;
        margin-bottom: -20px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="column"]:nth-of-type(3) div[data-testid="stPlotlyChart"] .js-plotly-plot {
        margin-bottom: -20px !important;
        padding-bottom: 0 !important;
    }
    div[data-testid="column"]:nth-of-type(2) div[data-testid="stPlotlyChart"] .js-plotly-plot {
        margin-top: 0px !important;
        padding-top: 0 !important;
    }
    div[data-testid="column"]:not(:nth-of-type(1)):not(:nth-of-type(2)):not(:nth-of-type(3)) div[data-testid="stPlotlyChart"] .js-plotly-plot {
        margin-top: -40px !important;
        margin-bottom: -10px !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        overflow: hidden !important;
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
    
    function styleCategoryPills() {
        document.querySelectorAll('button[key^="category_pill_"]').forEach(function(btn) {
            btn.style.setProperty('padding', '6px 12px', 'important');
            btn.style.setProperty('border-radius', '20px', 'important');
            btn.style.setProperty('background-color', 'white', 'important');
            btn.style.setProperty('color', 'black', 'important');
            btn.style.setProperty('font-family', 'Arial, sans-serif', 'important');
            btn.style.setProperty('font-size', '12px', 'important');
            btn.style.setProperty('cursor', 'pointer', 'important');
            btn.style.setProperty('margin', '0', 'important');
            btn.style.setProperty('transition', 'border 0.2s', 'important');
            btn.style.setProperty('width', '100%', 'important');
            var isSelected = btn.getAttribute('data-selected') === 'true';
            btn.style.setProperty('border', isSelected ? '2px solid black' : 'none', 'important');
            btn.style.setProperty('font-weight', isSelected ? 'bold' : 'normal', 'important');
        });
    }
    
    function init() {
        enforceFixedWidth();
        formatDateInput();
        styleCategoryPills();
    }
    
    init();
    window.addEventListener('load', function() { setTimeout(init, 500); });
    
    var observer = new MutationObserver(function() {
        setTimeout(formatDateInput, 100);
        setTimeout(styleCategoryPills, 100);
    });
    if (document.body) {
        observer.observe(document.body, { childList: true, subtree: true, attributes: true });
    }
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
def load_all_categories():
    """Load all categories from macro_consolidated table."""
    try:
        query = f"""
        SELECT DISTINCT category
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE category IS NOT NULL
        ORDER BY category ASC
        """
        df_categories = pandas_gbq.read_gbq(query, project_id=project_id, credentials=credentials, use_bqstorage_api=True)
        return df_categories['category'].tolist() if 'category' in df_categories.columns else []
    except Exception as e:
        st.error(f"Error loading categories: {e}")
        return []

@st.cache_data(ttl=3600)
def load_data(category_filter=None):
    """Load series from macro_consolidated table, optionally filtered by category."""
    try:
        if category_filter:
            query = f"""
            SELECT date, value, display_value, metric_name, units, category, display_format, percentile_desired, primary_source, target_min, target_max
            FROM `{project_id}.{dataset_id}.{table_id}`
            WHERE category = '{category_filter}'
            ORDER BY metric_name, date ASC
            """
        else:
            query = f"""
            SELECT date, value, display_value, metric_name, units, category, display_format, percentile_desired, primary_source, target_min, target_max
            FROM `{project_id}.{dataset_id}.{table_id}`
            ORDER BY category, metric_name, date ASC
            """
        df = pandas_gbq.read_gbq(query, project_id=project_id, credentials=credentials, use_bqstorage_api=True)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

with st.spinner("Loading data from BigQuery..."):
    all_categories = load_all_categories()
    df_all = load_data()
    selected_category = st.session_state.get('selected_category', 
        'Consumer Health' if 'Consumer Health' in all_categories else (all_categories[0] if all_categories else None))
    df = load_data(selected_category) if selected_category else pd.DataFrame()

if df_all.empty:
    st.warning("No data found.")
else:
    percentile_period = '15yr'
    start_date = None
    
    if all_categories:
        sorted_categories = sort_categories(all_categories)
    else:
        sorted_categories = []
    
    if sorted_categories:
        max_series = 0
        category_data = {}
        for cat in sorted_categories:
            category_df = df_all[df_all['category'] == cat].copy() if 'category' in df_all.columns else pd.DataFrame()
            if category_df.empty:
                continue
            unique_metrics = category_df['metric_name'].unique() if 'metric_name' in category_df.columns else []
            unique_metrics = filter_metrics(unique_metrics)
            num_series = len(unique_metrics)
            if num_series > 0:
                category_data[cat] = {
                    'df': category_df,
                    'metrics': sorted(unique_metrics),
                    'num_series': num_series
                }
                max_series = max(max_series, num_series)
        
        categories_with_data = [cat for cat in sorted_categories if cat in category_data]
        for idx, cat in enumerate(categories_with_data):
            cat_info = category_data[cat]
            cols = st.columns([1] + [1] * max_series)
            
            with cols[0]:
                st.markdown("<div style='height: 35px; padding-top: 0px; margin-top: 40px;'></div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; width: 100%;'><p style='font-size: 18px; font-family: Arial, sans-serif; margin: 0; font-weight: bold;'><u>{cat}</u></p></div>", unsafe_allow_html=True)
            
            for col_idx, metric_name in enumerate(cat_info['metrics'], start=1):
                if col_idx > max_series:
                    break
                df_metric = cat_info['df'][cat_info['df']['metric_name'] == metric_name].copy()
                result = process_metric_gauge_only(df_metric, percentile_period, start_date)
                
                if result:
                    with cols[col_idx]:
                        as_of_html = f"<p style='font-size: 10px; font-family: Arial, sans-serif; margin-top: 0; margin-bottom: 0; padding-top: 0; padding-bottom: 0; color: #666;'>{result.get('as_of_text', '')}</p>" if result.get('as_of_text') else ""
                        st.markdown(f"<div style='text-align: center; margin-bottom: 0px; padding-bottom: 0; margin-top: 10px; padding-top: 25px;'><p style='font-size: 12px; font-family: Arial, sans-serif; margin-bottom: 0; margin-top: 0; padding-bottom: 0;'><b>{result['metric_name']}</b></p>{as_of_html}</div>", unsafe_allow_html=True)
                        st.markdown("<div style='margin-top: -20px; padding-top: 0; display: flex; justify-content: center; align-items: center;'></div>", unsafe_allow_html=True)
                        st.plotly_chart(result['gauge_fig'], use_container_width=False, config=CHART_CONFIG, key=f"gauge_{cat}_{metric_name}")
            
            for col_idx in range(len(cat_info['metrics']) + 1, max_series + 1):
                with cols[col_idx]:
                    st.empty()
            
            if idx < len(categories_with_data) - 1:
                st.markdown("<hr style='border: none; border-top: 1px solid #ccc; margin-top: 30px; margin-bottom: -40px; padding-bottom: 20px; height: 0; width: 100%;'>", unsafe_allow_html=True)
                
    if not df.empty:
        category = get_column_value(df, 'category')
        if category:
            sorted_categories_for_filter = sort_categories(all_categories)
            
            st.markdown(f"<div id='category-section' style='background-color: black; color: white; padding-left: 5px; width: 100%; margin-bottom: 0.2rem; margin-top: 3rem;'><p style='font-size: 20px; font-family: Arial, sans-serif; margin: 0; text-align: left;'><b>Category Breakdown</b></p></div>", unsafe_allow_html=True)
            st.markdown(f"<div id='category-section' margin-bottom: 1.5rem; margin-top: 0.5rem;'><p style='font-size: 16px; font-family: Arial, sans-serif; margin: 0; text-align: left;'><i>Select a Category to Filter</i></p></div>", unsafe_allow_html=True)
            
            cols_pills = st.columns(len(sorted_categories_for_filter))
            for idx, cat_option in enumerate(sorted_categories_for_filter):
                is_selected = cat_option == category
                with cols_pills[idx]:
                    if st.button(cat_option, key=f"category_pill_{cat_option}", use_container_width=True):
                        st.session_state.selected_category = cat_option
                        st.rerun()
                    if is_selected:
                        st.markdown(f"""
                        <script>
                            setTimeout(function() {{
                                var btn = document.querySelector('button[key="category_pill_{cat_option}"]');
                                if (btn) {{
                                    btn.setAttribute('data-selected', 'true');
                                    btn.style.setProperty('border', '2px solid black', 'important');
                                }}
                            }}, 100);
                        </script>
                        """, unsafe_allow_html=True)
                  
            if st.session_state.get('scroll_to_bottom', False):
                st.markdown("""
                <script>
                    (function() {
                        function scrollToSection() {
                            var element = document.getElementById('category-section');
                            if (element) {
                                var rect = element.getBoundingClientRect();
                                var scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                                var targetY = rect.top + scrollTop - 20;
                                window.scrollTo({ top: targetY, behavior: 'smooth' });
                                return true;
                            }
                            return false;
                        }
                        if (!scrollToSection()) {
                            setTimeout(scrollToSection, 100);
                            setTimeout(scrollToSection, 300);
                            setTimeout(scrollToSection, 600);
                            setTimeout(scrollToSection, 1000);
                            setTimeout(scrollToSection, 1500);
                        }
                    })();
                </script>
                """, unsafe_allow_html=True)
                st.session_state.scroll_to_bottom = False
        
        if 'date' in df_all.columns and not df_all.empty:
            max_date = df_all['date'].max()
            min_date = df_all['date'].min()
            
            def to_date(d):
                return d.date() if isinstance(d, pd.Timestamp) else d
            
            min_date_input = to_date(min_date)
            max_date_input = to_date(max_date)
            default_start_date = pd.Timestamp('2010-01-01').date()
            if default_start_date < min_date_input:
                default_start_date = min_date_input
            elif default_start_date > max_date_input:
                default_start_date = max_date_input
            
            col_text, col_input = st.columns([0.85, 0.15])
            with col_text:
                st.markdown("<p style='margin: 0; padding: 0; margin-top: 0.3rem; text-align: right; font-size: 14px; font-family: Arial, sans-serif; font-style: italic; white-space: nowrap;'>Chart Start Date:</p>", unsafe_allow_html=True)
            with col_input:
                start_date = st.date_input("", value=default_start_date, min_value=min_date_input, max_value=max_date_input, key="start_date_filter", label_visibility="collapsed")
                    
        st.markdown(f"<div style='margin-top: 0.5rem; margin-bottom: 0.5rem;'><p style='font-size: 18px; font-family: Arial, sans-serif; margin: 0; text-align: center;'><b><u>{category}</u></b></p></div>", unsafe_allow_html=True)

        unique_metrics = df['metric_name'].unique() if 'metric_name' in df.columns else []
        unique_metrics = filter_metrics(unique_metrics)
        
        for idx, metric_name in enumerate(sorted(unique_metrics)):
            if idx > 0:
                st.markdown("<div style='margin-top: 1rem; padding-top: 1rem;'></div>", unsafe_allow_html=True)
            
            df_metric = df[df['metric_name'] == metric_name].copy()
            result = process_metric(df_metric, percentile_period, start_date)
            
            if result:
                col1, col2, col3 = st.columns([1, 1, 2.5])
                with col1:
                    if result['metric_name']:
                        source_text = f"<p style='font-size: 14px; font-family: Arial, sans-serif; margin-top: 0rem; margin-bottom: 0;'>Source: {result['primary_source']}</p>" if result['primary_source'] else ""
                        st.markdown(f"<div style='text-align: center; padding-top: 3.5rem;'><p style='font-size: 18px; font-family: Arial, sans-serif; margin: 0;'><b>{result['metric_name']}</b></p>{source_text}</div>", unsafe_allow_html=True)
                with col2:
                    st.markdown("<div style='padding-top: 1rem; display: flex; justify-content: center; align-items: center;'></div>", unsafe_allow_html=True)
                    st.plotly_chart(result['gauge_fig'], use_container_width=False, config=CHART_CONFIG, key=f"gauge_detail_{metric_name}")
                with col3:
                    st.plotly_chart(result['fig'], use_container_width=False, config=CHART_CONFIG, key=f"line_{metric_name}")
            
            st.markdown("<div style='margin-bottom: 3rem; padding-bottom: 1rem;'></div>", unsafe_allow_html=True)

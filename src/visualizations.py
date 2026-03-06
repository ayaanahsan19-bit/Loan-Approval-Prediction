"""
Visualization module for loan approval prediction app.
Contains all plotting functions using Plotly for interactive charts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
from typing import List, Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt


def create_kpi_card(title: str, value: str, subtitle: str = None) -> go.Figure:
    """
    Create a KPI card visualization.
    
    Args:
        title: Title of the KPI
        value: Main value to display
        subtitle: Optional subtitle
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="number+gauge+delta",
        value=float(value) if value.replace('.', '').isdigit() else 0,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#F5A623"},
            'steps': [
                {'range': [0, 50], 'color': "#1a1a1a"},
                {'range': [50, 100], 'color': "#2a2a2a"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=200,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D"
    )
    
    return fig


def create_donut_chart(data: pd.Series, title: str) -> go.Figure:
    """
    Create an interactive donut chart.
    
    Args:
        data: Series with values to plot
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[go.Pie(
        labels=data.index,
        values=data.values,
        hole=0.6,
        marker_colors=['#F5A623', '#1a1a1a', '#333333'],
        textinfo='label+percent',
        textfont_size=12
    )])
    
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D",
        font=dict(color="white")
    )
    
    return fig


def create_missing_values_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing missing values.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Plotly figure
    """
    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ['Column', 'Missing Count']
    missing_data['Missing Percentage'] = (missing_data['Missing Count'] / len(df)) * 100
    
    # Create heatmap
    fig = px.imshow(
        df.isnull().T,
        color_continuous_scale=['#1a1a1a', '#F5A623'],
        title="Missing Values Heatmap",
        labels=dict(x="Row Index", y="Columns", color="Missing")
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D",
        font=dict(color="white")
    )
    
    return fig


def create_distribution_plot(df: pd.DataFrame, column: str, color_by: str = None) -> go.Figure:
    """
    Create a distribution plot for numerical columns.
    
    Args:
        df: DataFrame containing the data
        column: Column name to plot
        color_by: Optional column to color by
        
    Returns:
        Plotly figure
    """
    if color_by and color_by in df.columns:
        fig = px.histogram(
            df, 
            x=column, 
            color=color_by,
            nbins=50,
            marginal="box",
            title=f"Distribution of {column} by {color_by}"
        )
    else:
        fig = px.histogram(
            df, 
            x=column, 
            nbins=50,
            marginal="box",
            title=f"Distribution of {column}"
        )
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D",
        font=dict(color="white")
    )
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create a correlation heatmap for numerical features.
    
    Args:
        df: DataFrame with numerical features
        
    Returns:
        Plotly figure
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numerical_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Heatmap",
        labels=dict(x="Features", y="Features", color="Correlation")
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=500,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D",
        font=dict(color="white")
    )
    
    return fig


def create_scatter_plot(df: pd.DataFrame, x: str, y: str, color_by: str = None) -> go.Figure:
    """
    Create a scatter plot.
    
    Args:
        df: DataFrame containing the data
        x: X-axis column name
        y: Y-axis column name
        color_by: Optional column to color points by
        
    Returns:
        Plotly figure
    """
    if color_by and color_by in df.columns:
        fig = px.scatter(
            df, 
            x=x, 
            y=y, 
            color=color_by,
            title=f"{y} vs {x} by {color_by}",
            hover_data=[col for col in df.columns if col not in [x, y, color_by]]
        )
    else:
        fig = px.scatter(
            df, 
            x=x, 
            y=y,
            title=f"{y} vs {x}",
            hover_data=[col for col in df.columns if col not in [x, y]]
        )
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D",
        font=dict(color="white")
    )
    
    return fig


def create_model_comparison_chart(results: Dict[str, Any]) -> go.Figure:
    """
    Create a comparison chart for model performance.
    
    Args:
        results: Dictionary containing model results
        
    Returns:
        Plotly figure
    """
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    
    fig = go.Figure()
    
    for i, model in enumerate(models):
        values = [results[model].get(metric, 0) for metric in metrics]
        values = [v * 100 if isinstance(v, (int, float)) and v <= 1 else v for v in values]
        
        fig.add_trace(go.Bar(
            name=model.replace('_', ' ').title(),
            x=metrics,
            y=values,
            marker_color='#F5A623' if i == 0 else '#FF6B6B'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Metrics",
        yaxis_title="Score (%)",
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D",
        font=dict(color="white"),
        barmode='group'
    )
    
    return fig


def create_feature_importance_plot(feature_importance: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Create a horizontal bar chart for feature importance.
    
    Args:
        feature_importance: DataFrame with feature importance
        top_n: Number of top features to show
        
    Returns:
        Plotly figure
    """
    top_features = feature_importance.head(top_n)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f"Top {top_n} Feature Importance",
        color='importance',
        color_continuous_scale='YlOrRd'
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D",
        font=dict(color="white"),
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_probability_gauge(probability: float) -> go.Figure:
    """
    Create a gauge chart for prediction probability.
    
    Args:
        probability: Prediction probability (0-1)
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Approval Probability"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "#F5A623"},
            'steps': [
                {'range': [0, 25], 'color': "#1a1a1a"},
                {'range': [25, 50], 'color': "#2a2a2a"},
                {'range': [50, 75], 'color': "#3a3a3a"},
                {'range': [75, 100], 'color': "#4a4a4a"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D",
        font=dict(color="white")
    )
    
    return fig


def create_approval_badge(is_approved: bool, probability: float) -> go.Figure:
    """
    Create a visual badge for approval/rejection.
    
    Args:
        is_approved: Whether loan is approved
        probability: Approval probability
        
    Returns:
        Plotly figure
    """
    status = "APPROVED ✅" if is_approved else "REJECTED ❌"
    color = "#4CAF50" if is_approved else "#F44336"
    
    fig = go.Figure()
    
    fig.add_annotation(
        x=0.5, y=0.5,
        text=f"<span style='font-size: 24px; font-weight: bold; color: {color}'>{status}</span><br>"
             f"<span style='font-size: 16px; color: white'>Confidence: {probability:.1%}</span>",
        showarrow=False,
        xref="paper", yref="paper",
        xanchor='middle', yanchor='middle'
    )
    
    fig.update_layout(
        template="plotly_dark",
        height=200,
        paper_bgcolor="#0D0D0D",
        plot_bgcolor="#0D0D0D",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False)
    )
    
    return fig

"""
Loan Approval Prediction Streamlit App
A comprehensive web application for loan approval prediction with ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import classification_report
import joblib
import os
import sys

def html(content: str):
    """Safe HTML renderer compatible with all Streamlit versions"""
    st.markdown(content, unsafe_allow_html=True)

# Set page configuration FIRST
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    try:
        with open("assets/style.css") as f:
            html(f"<style>{f.read()}</style>")
    except FileNotFoundError:
        # Fallback CSS if file not found
        html("""
        <style>
        .main { background-color: #0D0D0D; }
        .stApp { background-color: #0D0D0D; }
        </style>
        """)

# Load CSS
load_css()

# Add src directory to path for imports
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from src.preprocess import (
        load_data, clean_data, encode_categorical_features, 
        scale_features, prepare_features_and_target, get_preprocessing_info
    )
    from src.model import LoanApprovalModel
    from src.visualizations import (
        create_kpi_card, create_donut_chart, create_missing_values_heatmap,
        create_distribution_plot, create_correlation_heatmap, create_scatter_plot,
        create_model_comparison_chart, create_feature_importance_plot,
        create_probability_gauge, create_approval_badge
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    st.warning(f"Some modules not available: {e}")
    IMPORTS_AVAILABLE = False

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = LoanApprovalModel() if IMPORTS_AVAILABLE else None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None

# Sidebar navigation
st.sidebar.title("🏦 Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    [
        "🏠 Home / Overview",
        "🔍 Exploratory Data Analysis", 
        "⚙️ Data Preprocessing",
        "🤖 Model Training",
        "⚖️ Model Comparison",
        "🎯 Live Predictor"
    ]
)

# Load data function
@st.cache_data
def load_dataset():
    """Load the loan approval dataset."""
    try:
        data_path = "loan_approval_dataset.csv"
        if os.path.exists(data_path):
            if IMPORTS_AVAILABLE:
                return load_data(data_path)
            else:
                return pd.read_csv(data_path)
        else:
            st.error("Dataset not found! Please ensure 'loan_approval_dataset.csv' is in the root directory.")
            return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# PAGE 1: Home / Overview
if page == "🏠 Home / Overview":
    html('''
    <div class="hero-section">
        <h1 class="hero-title">🏦 Loan Approval Prediction System</h1>
        <p class="hero-subtitle">Advanced Machine Learning for Intelligent Loan Decision Making</p>
    </div>
    ''')
    
    if not IMPORTS_AVAILABLE:
        st.error("⚠️ Some modules are not available. Running in basic mode.")
        st.info("Please check your requirements.txt and ensure all dependencies are installed.")
    
    # Load data
    if st.session_state.data is None:
        with st.spinner("Loading dataset..."):
            st.session_state.data = load_dataset()
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            html(f'''
            <div class="metric-card">
                <div class="metric-value">{len(data):,}</div>
                <div class="metric-label">Total Applications</div>
            </div>
            ''')
        
        with col2:
            html(f'''
            <div class="metric-card">
                <div class="metric-value">{len(data.columns)}</div>
                <div class="metric-label">Features</div>
            </div>
            ''')
        
        # Calculate approval rate
        if 'loan_status' in data.columns:
            approval_rate = (data['loan_status'] == 'Approved').sum() / len(data) * 100
            with col3:
                html(f'''
            <div class="metric-card">
                <div class="metric-value">{approval_rate:.1f}%</div>
                <div class="metric-label">Approval Rate</div>
            </div>
            ''')
        
        with col4:
            missing_count = data.isnull().sum().sum()
            html(f'''
            <div class="metric-card">
                <div class="metric-value">{missing_count}</div>
                <div class="metric-label">Missing Values</div>
            </div>
            ''')
        
        # Class imbalance visualization
        if 'loan_status' in data.columns:
            st.markdown("### 📊 Loan Status Distribution")
            
            if IMPORTS_AVAILABLE:
                # Create donut chart
                loan_status_counts = data['loan_status'].value_counts()
                fig = create_donut_chart(loan_status_counts, "Loan Status Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback: show simple bar chart
                loan_status_counts = data['loan_status'].value_counts()
                st.bar_chart(loan_status_counts)
            
            # Show class imbalance note
            st.info("📌 **Note**: The dataset shows class imbalance, which is common in loan approval scenarios. We'll address this using SMOTE in the model training phase.")
        
        # Sample data preview
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Dataset info
        st.markdown("### 📈 Dataset Information")
        if IMPORTS_AVAILABLE:
            info = get_preprocessing_info(data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Data Types:**")
                for col, dtype in info['data_types'].items():
                    st.write(f"• {col}: {dtype}")
            
            with col2:
                st.markdown("**Missing Values:**")
                for col, missing in info['missing_values'].items():
                    if missing > 0:
                        st.write(f"• {col}: {missing}")
        else:
            st.info("Dataset information available in full mode.")
    else:
        st.warning("Please upload your dataset to continue.")

# PAGE 2: Exploratory Data Analysis
elif page == "🔍 Exploratory Data Analysis":
    html('<h1 class="hero-title">🔍 Exploratory Data Analysis</h1>')
    
    if not IMPORTS_AVAILABLE:
        st.error("⚠️ Advanced features not available. Please check dependencies.")
        st.info("This page requires full module imports to function properly.")
        st.stop()
    
    if st.session_state.data is None:
        st.session_state.data = load_dataset()
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Sidebar filters
        st.sidebar.markdown("### 🔎 Interactive Filters")
        
        # Filter by education
        if 'education' in data.columns:
            education_filter = st.sidebar.multiselect(
                "Education Level",
                options=data['education'].unique(),
                default=data['education'].unique()
            )
            data = data[data['education'].isin(education_filter)]
        
        # Filter by self employed
        if 'self_employed' in data.columns:
            self_employed_filter = st.sidebar.multiselect(
                "Self Employed",
                options=data['self_employed'].unique(),
                default=data['self_employed'].unique()
            )
            data = data[data['self_employed'].isin(self_employed_filter)]
        
        # Missing values heatmap
        st.markdown("### 🔥 Missing Values Heatmap")
        with st.spinner("Generating missing values heatmap..."):
            fig = create_missing_values_heatmap(data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots for numerical features
        st.markdown("### 📊 Feature Distributions")
        
        numerical_cols = ['income_annum', 'loan_amount', 'cibil_score', 
                         'residential_assets_value', 'commercial_assets_value', 
                         'luxury_assets_value', 'bank_asset_value']
        
        numerical_cols = [col for col in numerical_cols if col in data.columns]
        
        if numerical_cols:
            selected_col = st.selectbox("Select feature to visualize", numerical_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                # Distribution plot
                fig = create_distribution_plot(data, selected_col, 'loan_status' if 'loan_status' in data.columns else None)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot by loan status
                if 'loan_status' in data.columns:
                    fig = px.box(data, x='loan_status', y=selected_col, 
                               title=f"{selected_col} by Loan Status",
                               color='loan_status')
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔗 Feature Correlations")
        with st.spinner("Computing correlations..."):
            fig = create_correlation_heatmap(data)
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plots
        st.markdown("### 📈 Relationship Analysis")
        
        if 'income_annum' in data.columns and 'loan_amount' in data.columns:
            fig = create_scatter_plot(data, 'income_annum', 'loan_amount', 
                                    'loan_status' if 'loan_status' in data.columns else None)
            st.plotly_chart(fig, use_container_width=True)
        
        if 'cibil_score' in data.columns and 'loan_amount' in data.columns:
            fig = create_scatter_plot(data, 'cibil_score', 'loan_amount', 
                                    'loan_status' if 'loan_status' in data.columns else None)
            st.plotly_chart(fig, use_container_width=True)

# Simplified fallback pages
elif page == "⚙️ Data Preprocessing":
    html('<h1 class="hero-title">⚙️ Data Preprocessing</h1>')
    if not IMPORTS_AVAILABLE:
        st.error("⚠️ This feature requires full module imports.")
        st.info("Please ensure all dependencies are properly installed.")
        st.stop()
    st.info("Data preprocessing features available in full mode.")

elif page == "🤖 Model Training":
    html('<h1 class="hero-title">🤖 Model Training</h1>')
    if not IMPORTS_AVAILABLE:
        st.error("⚠️ Model training requires full module imports.")
        st.info("Please ensure all dependencies are properly installed.")
        st.stop()
    st.info("Model training features available in full mode.")

elif page == "⚖️ Model Comparison":
    html('<h1 class="hero-title">⚖️ Model Comparison</h1>')
    if not IMPORTS_AVAILABLE:
        st.error("⚠️ Model comparison requires full module imports.")
        st.info("Please ensure all dependencies are properly installed.")
        st.stop()
    st.info("Model comparison features available in full mode.")

elif page == "🎯 Live Predictor":
    html('<h1 class="hero-title">🎯 Live Loan Approval Predictor</h1>')
    if not IMPORTS_AVAILABLE:
        st.error("⚠️ Live prediction requires full module imports.")
        st.info("Please ensure all dependencies are properly installed.")
        st.stop()
    st.info("Live prediction features available in full mode.")

# Footer
st.markdown("---")
html("""
<div style="text-align: center; color: #B0B0B0; padding: 1rem;">
    <p>🏦 Loan Approval Prediction System | Built with Streamlit & Machine Learning</p>
    <p>© 2024 | Advanced Analytics for Financial Decision Making</p>
</div>
""")

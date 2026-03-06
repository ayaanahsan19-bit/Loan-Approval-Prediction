"""
Simple Loan Approval Prediction App - Ultra Compatible Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# Set page configuration FIRST - this must be the very first st command
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple CSS without file dependencies
st.markdown("""
<style>
.main {
    background-color: #0D0D0D;
    color: white;
}
.stApp {
    background-color: #0D0D0D;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #F5A623;
    margin: 0.5rem 0;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: #F5A623;
}
.metric-label {
    color: #B0B0B0;
    margin-top: 0.5rem;
}
.hero-section {
    background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
    padding: 2rem;
    border-radius: 15px;
    border: 2px solid #F5A623;
    margin: 1rem 0;
    text-align: center;
}
.hero-title {
    font-size: 2.5rem;
    color: #F5A623;
    margin-bottom: 1rem;
}
.hero-subtitle {
    font-size: 1.2rem;
    color: #B0B0B0;
}
</style>
""", unsafe_allow_html=True)

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
            return pd.read_csv(data_path)
        else:
            st.error("Dataset not found! Please ensure 'loan_approval_dataset.csv' is in the root directory.")
            return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# PAGE 1: Home / Overview
if page == "🏠 Home / Overview":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🏦 Loan Approval Prediction System</h1>
        <p class="hero-subtitle">Advanced Machine Learning for Intelligent Loan Decision Making</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    if st.session_state.data is None:
        with st.spinner("Loading dataset..."):
            st.session_state.data = load_dataset()
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(data):,}</div>
                <div class="metric-label">Total Applications</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(data.columns)}</div>
                <div class="metric-label">Features</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Calculate approval rate
        if 'loan_status' in data.columns:
            approval_rate = (data['loan_status'] == 'Approved').sum() / len(data) * 100
            with col3:
                st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{approval_rate:.1f}%</div>
                <div class="metric-label">Approval Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            missing_count = data.isnull().sum().sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{missing_count}</div>
                <div class="metric-label">Missing Values</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Class imbalance visualization
        if 'loan_status' in data.columns:
            st.markdown("### 📊 Loan Status Distribution")
            
            # Simple bar chart
            loan_status_counts = data['loan_status'].value_counts()
            fig = px.bar(loan_status_counts, title="Loan Status Distribution", 
                        color=loan_status_counts.index, template="plotly_dark")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show class imbalance note
            st.info("📌 **Note**: The dataset shows class imbalance, which is common in loan approval scenarios.")
        
        # Sample data preview
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Dataset info
        st.markdown("### 📈 Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Types:**")
            for col, dtype in data.dtypes.items():
                st.write(f"• {col}: {dtype}")
        
        with col2:
            st.markdown("**Missing Values:**")
            missing_vals = data.isnull().sum()
            for col, missing in missing_vals.items():
                if missing > 0:
                    st.write(f"• {col}: {missing}")
    else:
        st.warning("Please upload your dataset to continue.")

# Simple fallback pages
elif page == "🔍 Exploratory Data Analysis":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🔍 Exploratory Data Analysis</h1>
    </div>
    """, unsafe_allow_html=True)
    st.info("Advanced EDA features coming soon! For now, check the Home page for data overview.")

elif page == "⚙️ Data Preprocessing":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">⚙️ Data Preprocessing</h1>
    </div>
    """, unsafe_allow_html=True)
    st.info("Data preprocessing features coming soon!")

elif page == "🤖 Model Training":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🤖 Model Training</h1>
    </div>
    """, unsafe_allow_html=True)
    st.info("Model training features coming soon!")

elif page == "⚖️ Model Comparison":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">⚖️ Model Comparison</h1>
    </div>
    """, unsafe_allow_html=True)
    st.info("Model comparison features coming soon!")

elif page == "🎯 Live Predictor":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🎯 Live Loan Approval Predictor</h1>
    </div>
    """, unsafe_allow_html=True)
    st.info("Live prediction features coming soon!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #B0B0B0; padding: 1rem;">
    <p>🏦 Loan Approval Prediction System | Built with Streamlit & Machine Learning</p>
    <p>© 2024 | Advanced Analytics for Financial Decision Making</p>
</div>
""", unsafe_allow_html=True)

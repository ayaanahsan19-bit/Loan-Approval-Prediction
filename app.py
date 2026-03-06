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

# Add src directory to path for imports
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

# Load custom CSS
def load_css():
    with open('assets/style.css', 'r') as f:
        css = f.read()
    st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
load_css()

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = LoanApprovalModel()
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
            return load_data(data_path)
        else:
            st.error("Dataset not found! Please ensure 'loan_approval_dataset.csv' is in the root directory.")
            return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

# PAGE 1: Home / Overview
if page == "🏠 Home / Overview":
    st.markdown('<div class="hero-section">', unsafe_allow_allow_html=True)
    st.markdown('<h1 class="hero-title">🏦 Loan Approval Prediction System</h1>', unsafe_allow_allow_html=True)
    st.markdown('<p class="hero-subtitle">Advanced Machine Learning for Intelligent Loan Decision Making</p>', unsafe_allow_allow_html=True)
    st.markdown('</div>', unsafe_allow_allow_html=True)
    
    # Load data
    if st.session_state.data is None:
        with st.spinner("Loading dataset..."):
            st.session_state.data = load_dataset()
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Dataset statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_allow_html=True)
            st.markdown(f'<div class="metric-value">{len(data):,}</div>', unsafe_allow_allow_html=True)
            st.markdown('<div class="metric-label">Total Applications</div>', unsafe_allow_allow_html=True)
            st.markdown('</div>', unsafe_allow_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_allow_html=True)
            st.markdown(f'<div class="metric-value">{len(data.columns)}</div>', unsafe_allow_allow_html=True)
            st.markdown('<div class="metric-label">Features</div>', unsafe_allow_allow_html=True)
            st.markdown('</div>', unsafe_allow_allow_html=True)
        
        # Calculate approval rate
        if 'loan_status' in data.columns:
            approval_rate = (data['loan_status'] == 'Approved').sum() / len(data) * 100
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_allow_html=True)
                st.markdown(f'<div class="metric-value">{approval_rate:.1f}%</div>', unsafe_allow_allow_html=True)
                st.markdown('<div class="metric-label">Approval Rate</div>', unsafe_allow_allow_html=True)
                st.markdown('</div>', unsafe_allow_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_allow_html=True)
            missing_count = data.isnull().sum().sum()
            st.markdown(f'<div class="metric-value">{missing_count}</div>', unsafe_allow_allow_html=True)
            st.markdown('<div class="metric-label">Missing Values</div>', unsafe_allow_allow_html=True)
            st.markdown('</div>', unsafe_allow_allow_html=True)
        
        # Class imbalance visualization
        if 'loan_status' in data.columns:
            st.markdown("### 📊 Loan Status Distribution")
            
            # Create donut chart
            loan_status_counts = data['loan_status'].value_counts()
            fig = create_donut_chart(loan_status_counts, "Loan Status Distribution")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show class imbalance note
            st.info("📌 **Note**: The dataset shows class imbalance, which is common in loan approval scenarios. We'll address this using SMOTE in the model training phase.")
        
        # Sample data preview
        st.markdown("### 📋 Dataset Preview")
        st.dataframe(data.head(10), use_container_width=True)
        
        # Dataset info
        st.markdown("### 📈 Dataset Information")
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

# PAGE 2: Exploratory Data Analysis
elif page == "🔍 Exploratory Data Analysis":
    st.markdown('<h1 class="hero-title">🔍 Exploratory Data Analysis</h1>', unsafe_allow_allow_html=True)
    
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

# PAGE 3: Data Preprocessing
elif page == "⚙️ Data Preprocessing":
    st.markdown('<h1 class="hero-title">⚙️ Data Preprocessing</h1>', unsafe_allow_allow_html=True)
    
    if st.session_state.data is None:
        st.session_state.data = load_dataset()
    
    if st.session_state.data is not None:
        raw_data = st.session_state.data
        
        st.markdown("### 🧹 Data Cleaning Process")
        
        # Show raw vs cleaned data side by side
        with st.spinner("Cleaning data..."):
            cleaned_data = clean_data(raw_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Raw Data**")
            st.dataframe(raw_data.head(), use_container_width=True)
            st.write(f"Shape: {raw_data.shape}")
        
        with col2:
            st.markdown("**Cleaned Data**")
            st.dataframe(cleaned_data.head(), use_container_width=True)
            st.write(f"Shape: {cleaned_data.shape}")
        
        # Explain imputation strategies
        st.markdown("### 📝 Imputation Strategies")
        
        with st.expander("🔢 Numerical Features Imputation"):
            st.write("""
            **Strategy**: Median Imputation
            - **Why Median?**: Robust to outliers and skewed distributions
            - **Features Applied**: income_annum, loan_amount, cibil_score, residential_assets_value, 
              commercial_assets_value, luxury_assets_value, bank_asset_value, loan_term
            - **Impact**: Preserves central tendency while being resistant to extreme values
            """)
        
        with st.expander("📊 Categorical Features Imputation"):
            st.write("""
            **Strategy**: Mode Imputation
            - **Why Mode?**: Preserves the most frequent category
            - **Features Applied**: education, self_employed
            - **Impact**: Maintains the distribution of categorical variables
            """)
        
        # Encoding explanation
        st.markdown("### 🏷️ Categorical Encoding")
        
        with st.spinner("Encoding categorical features..."):
            encoded_data, encoders = encode_categorical_features(cleaned_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Before Encoding**")
            st.dataframe(cleaned_data[['education', 'self_employed', 'loan_status']].head(), 
                        use_container_width=True)
        
        with col2:
            st.markdown("**After Encoding**")
            st.dataframe(encoded_data[['education', 'self_employed', 'loan_status']].head(), 
                        use_container_width=True)
        
        # Show encoding mappings
        st.markdown("**Encoding Mappings:**")
        for feature, encoder in encoders.items():
            if hasattr(encoder, 'classes_'):
                mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                st.write(f"• {feature}: {mapping}")
        
        # Feature scaling
        st.markdown("### ⚖️ Feature Scaling")
        
        with st.spinner("Scaling features..."):
            scaled_data, scaler = scale_features(encoded_data)
        
        st.markdown("**StandardScaler Applied**")
        st.write("""
        - **Purpose**: Standardize features by removing mean and scaling to unit variance
        - **Formula**: z = (x - μ) / σ
        - **Impact**: Ensures all features contribute equally to model training
        """)
        
        # Store preprocessed data
        st.session_state.preprocessed_data = scaled_data
        
        # Download cleaned dataset
        st.markdown("### 💾 Download Cleaned Dataset")
        
        csv = cleaned_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv,
            file_name="cleaned_loan_data.csv",
            mime="text/csv"
        )
        
        st.success("✅ Data preprocessing completed successfully!")

# PAGE 4: Model Training
elif page == "🤖 Model Training":
    st.markdown('<h1 class="hero-title">🤖 Model Training</h1>', unsafe_allow_allow_html=True)
    
    if st.session_state.preprocessed_data is None:
        st.error("Please complete data preprocessing first!")
    else:
        # Sidebar controls
        st.sidebar.markdown("### ⚙️ Training Parameters")
        
        test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.sidebar.slider("Random State", 0, 100, 42)
        apply_smote = st.sidebar.checkbox("Apply SMOTE", value=True)
        
        model_choice = st.sidebar.selectbox(
            "Choose Model",
            ["Logistic Regression", "Decision Tree", "Both"]
        )
        
        # Train button
        if st.sidebar.button("🚀 Start Training", type="primary"):
            with st.spinner("Preparing data..."):
                # Prepare features and target
                X, y = prepare_features_and_target(st.session_state.preprocessed_data)
                
                # Initialize model
                model = LoanApprovalModel()
            
            with st.spinner("Training models..."):
                # Train and evaluate
                results = model.train_and_evaluate_models(
                    X, y, test_size=test_size, random_state=random_state, 
                    apply_smote_flag=apply_smote
                )
                
                # Store results
                st.session_state.model_results = results
                st.session_state.model = model
            
            st.success("🎉 Model training completed!")
        
        # Display results if available
        if st.session_state.model_results is not None:
            results = st.session_state.model_results
            
            # Progress bar for training completion
            st.progress(1.0, text="Training Complete!")
            
            # Display results for each model
            if 'logistic_regression' in results and (model_choice == "Logistic Regression" or model_choice == "Both"):
                st.markdown("### 📊 Logistic Regression Results")
                
                lr_results = results['logistic_regression']
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{lr_results['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{lr_results['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{lr_results['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{lr_results['f1_score']:.3f}")
                
                # Classification report
                st.markdown("**Classification Report:**")
                report_df = pd.DataFrame(lr_results['classification_report']).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
                
                # Confusion Matrix
                st.markdown("**Confusion Matrix:**")
                model = st.session_state.model
                model.results['logistic_regression']['y_test'] = results['train_test_split']['y_test']
                fig = model.create_confusion_matrix_plot('logistic_regression')
                st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve
                st.markdown("**ROC Curve:**")
                fig = model.create_roc_curve_plot('logistic_regression')
                st.plotly_chart(fig, use_container_width=True)
            
            if 'decision_tree' in results and (model_choice == "Decision Tree" or model_choice == "Both"):
                st.markdown("### 🌳 Decision Tree Results")
                
                dt_results = results['decision_tree']
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{dt_results['accuracy']:.3f}")
                with col2:
                    st.metric("Precision", f"{dt_results['precision']:.3f}")
                with col3:
                    st.metric("Recall", f"{dt_results['recall']:.3f}")
                with col4:
                    st.metric("F1-Score", f"{dt_results['f1_score']:.3f}")
                
                # Classification report
                st.markdown("**Classification Report:**")
                report_df = pd.DataFrame(dt_results['classification_report']).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
                
                # Confusion Matrix
                st.markdown("**Confusion Matrix:**")
                model = st.session_state.model
                model.results['decision_tree']['y_test'] = results['train_test_split']['y_test']
                fig = model.create_confusion_matrix_plot('decision_tree')
                st.plotly_chart(fig, use_container_width=True)
                
                # ROC Curve
                st.markdown("**ROC Curve:**")
                fig = model.create_roc_curve_plot('decision_tree')
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature Importance
                st.markdown("**Feature Importance:**")
                feature_names = X.columns.tolist()
                feature_importance = model.get_feature_importance('decision_tree', feature_names)
                fig = create_feature_importance_plot(feature_importance)
                st.plotly_chart(fig, use_container_width=True)
            
            # Save models
            if st.button("💾 Save Models"):
                st.session_state.model.save_models()
                st.success("Models saved successfully!")

# PAGE 5: Model Comparison
elif page == "⚖️ Model Comparison":
    st.markdown('<h1 class="hero-title">⚖️ Model Comparison</h1>', unsafe_allow_allow_html=True)
    
    if st.session_state.model_results is None:
        st.error("Please train models first!")
    else:
        results = st.session_state.model_results
        
        # Side-by-side comparison
        st.markdown("### 📊 Performance Comparison")
        
        comparison_data = {}
        for model_name in ['logistic_regression', 'decision_tree']:
            if model_name in results:
                comparison_data[model_name.replace('_', ' ').title()] = {
                    'Accuracy': results[model_name]['accuracy'] * 100,
                    'Precision': results[model_name]['precision'] * 100,
                    'Recall': results[model_name]['recall'] * 100,
                    'F1-Score': results[model_name]['f1_score'] * 100,
                    'ROC-AUC': results[model_name]['roc_auc'] * 100
                }
        
        if comparison_data:
            fig = create_model_comparison_chart(results)
            st.plotly_chart(fig, use_container_width=True)
            
            # Comparison table
            st.markdown("**Detailed Comparison Table:**")
            comparison_df = pd.DataFrame(comparison_data).transpose()
            st.dataframe(comparison_df.style.background_gradient(cmap='YlOrRd', axis=0), 
                        use_container_width=True)
            
            # SMOTE comparison
            st.markdown("### 🔄 SMOTE Impact Analysis")
            if results.get('smote_applied'):
                st.success("✅ SMOTE was applied to handle class imbalance")
                st.info("""
                **SMOTE Benefits:**
                - Improves minority class (loan rejections) detection
                - Better balance between precision and recall
                - More robust model performance
                """)
            else:
                st.warning("⚠️ SMOTE was not applied")
                st.info("""
                **Without SMOTE:**
                - Model may be biased toward majority class
                - Higher accuracy but poor minority class detection
                - Risk of overfitting to approval patterns
                """)
            
            # Winner declaration
            st.markdown("### 🏆 Model Winner")
            
            # Calculate overall scores
            lr_score = 0
            dt_score = 0
            
            if 'logistic_regression' in results:
                lr_score = (results['logistic_regression']['accuracy'] + 
                          results['logistic_regression']['precision'] + 
                          results['logistic_regression']['recall'] + 
                          results['logistic_regression']['f1_score'] + 
                          results['logistic_regression']['roc_auc']) / 5
            
            if 'decision_tree' in results:
                dt_score = (results['decision_tree']['accuracy'] + 
                          results['decision_tree']['precision'] + 
                          results['decision_tree']['recall'] + 
                          results['decision_tree']['f1_score'] + 
                          results['decision_tree']['roc_auc']) / 5
            
            if lr_score > dt_score:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #F5A623, #FFB84D); 
                           padding: 2rem; border-radius: 15px; text-align: center; 
                           color: #0D0D0D; margin: 2rem 0;">
                    <h2>🏆 Logistic Regression Wins!</h2>
                    <p>Best overall performance with balanced metrics</p>
                </div>
                """, unsafe_allow_allow_html=True)
            elif dt_score > lr_score:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #F5A623, #FFB84D); 
                           padding: 2rem; border-radius: 15px; text-align: center; 
                           color: #0D0D0D; margin: 2rem 0;">
                    <h2>🏆 Decision Tree Wins!</h2>
                    <p>Best overall performance with strong feature interpretability</p>
                </div>
                """, unsafe_allow_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #F5A623, #FFB84D); 
                           padding: 2rem; border-radius: 15px; text-align: center; 
                           color: #0D0D0D; margin: 2rem 0;">
                    <h2>🤝 It's a Tie!</h2>
                    <p>Both models perform equally well</p>
                </div>
                """, unsafe_allow_allow_html=True)

# PAGE 6: Live Predictor
elif page == "🎯 Live Predictor":
    st.markdown('<h1 class="hero-title">🎯 Live Loan Approval Predictor</h1>', unsafe_allow_allow_html=True)
    
    if st.session_state.model is None or not st.session_state.model.models:
        st.error("Please train models first!")
    else:
        st.markdown("### 📝 Enter Applicant Details")
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            no_of_dependents = st.number_input("Number of Dependents", 0, 10, 0)
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self Employed", ["Yes", "No"])
            income_annum = st.number_input("Annual Income", 0, 10000000, 500000, step=10000)
            loan_amount = st.number_input("Loan Amount", 0, 50000000, 1000000, step=50000)
        
        with col2:
            loan_term = st.number_input("Loan Term (months)", 1, 360, 12)
            cibil_score = st.number_input("CIBIL Score", 300, 900, 650)
            residential_assets_value = st.number_input("Residential Assets Value", 0, 10000000, 0, step=10000)
            commercial_assets_value = st.number_input("Commercial Assets Value", 0, 10000000, 0, step=10000)
            luxury_assets_value = st.number_input("Luxury Assets Value", 0, 10000000, 0, step=10000)
            bank_asset_value = st.number_input("Bank Asset Value", 0, 10000000, 0, step=10000)
        
        # Model selection
        model_choice = st.selectbox("Select Model for Prediction", 
                                  ["Logistic Regression", "Decision Tree"])
        
        # Predict button
        if st.button("🔮 Predict Loan Status", type="primary"):
            with st.spinner("Making prediction..."):
                # Create input dataframe
                input_data = pd.DataFrame({
                    'no_of_dependents': [no_of_dependents],
                    'education': [education],
                    'self_employed': [self_employed],
                    'income_annum': [income_annum],
                    'loan_amount': [loan_amount],
                    'loan_term': [loan_term],
                    'cibil_score': [cibil_score],
                    'residential_assets_value': [residential_assets_value],
                    'commercial_assets_value': [commercial_assets_value],
                    'luxury_assets_value': [luxury_assets_value],
                    'bank_asset_value': [bank_asset_value]
                })
                
                # Apply same preprocessing
                input_cleaned = clean_data(input_data)
                input_encoded, _ = encode_categorical_features(input_cleaned)
                
                # Get feature order from training data
                if st.session_state.preprocessed_data is not None:
                    X, _ = prepare_features_and_target(st.session_state.preprocessed_data)
                    input_scaled = input_encoded[X.columns]
                    
                    # Apply scaling
                    if hasattr(st.session_state.model, 'scalers') and st.session_state.model.scalers:
                        scaler = list(st.session_state.model.scalers.values())[0]
                        numerical_cols = ['income_annum', 'loan_amount', 'cibil_score', 
                                        'residential_assets_value', 'commercial_assets_value', 
                                        'luxury_assets_value', 'bank_asset_value', 'loan_term']
                        numerical_cols = [col for col in numerical_cols if col in input_scaled.columns]
                        input_scaled[numerical_cols] = scaler.transform(input_scaled[numerical_cols])
                    
                    # Make prediction
                    model_name = 'logistic_regression' if model_choice == "Logistic Regression" else 'decision_tree'
                    prediction, probability = st.session_state.model.predict_single(model_name, input_scaled)
                    
                    # Display results
                    st.markdown("### 🎯 Prediction Result")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Approval badge
                        is_approved = bool(prediction)
                        fig = create_approval_badge(is_approved, probability)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Probability gauge
                        fig = create_probability_gauge(probability)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance
                    st.markdown("### 📊 Feature Influence")
                    feature_names = X.columns.tolist()
                    feature_importance = st.session_state.model.get_feature_importance(model_name, feature_names)
                    fig = create_feature_importance_plot(feature_importance, top_n=8)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendation
                    st.markdown("### 💡 Recommendation")
                    if is_approved:
                        st.success("✅ **LOAN APPROVED** - The applicant meets the criteria for loan approval based on our model analysis.")
                    else:
                        st.error("❌ **LOAN REJECTED** - The applicant does not meet the criteria for loan approval based on our model analysis.")
                        st.info("💡 **Suggestions for Improvement:**")
                        if cibil_score < 650:
                            st.write("• Improve CIBIL score by maintaining good credit history")
                        if income_annum < 1000000:
                            st.write("• Increase annual income through additional sources")
                        if loan_amount > income_annum * 10:
                            st.write("• Consider reducing loan amount to improve debt-to-income ratio")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #B0B0B0; padding: 1rem;">
    <p>🏦 Loan Approval Prediction System | Built with Streamlit & Machine Learning</p>
    <p>© 2024 | Advanced Analytics for Financial Decision Making</p>
</div>
""", unsafe_allow_allow_html=True)

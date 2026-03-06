"""
Simple Loan Approval Prediction App - Ultra Compatible Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

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
            df = pd.read_csv(data_path)
            df.columns = df.columns.str.strip()
            for _c in df.select_dtypes(include='object').columns:
                df[_c] = df[_c].str.strip()
            return df
        else:
            st.error("Dataset not found! Please ensure 'loan_approval_dataset.csv' is in the root directory.")
            return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


def preprocess_data(df):
    """Encode + scale raw df. Returns X, y, feature_cols, scaler, encoders."""
    data = df.copy()
    data.drop(columns=['loan_id'], errors='ignore', inplace=True)
    for col in data.select_dtypes(include=np.number).columns:
        data[col] = data[col].fillna(data[col].median())
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].fillna(data[col].mode()[0])
    encoders = {}
    for col in ['education', 'self_employed', 'loan_status']:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            encoders[col] = le
    feature_cols = [c for c in data.columns if c != 'loan_status']
    X = data[feature_cols].copy()
    y = data['loan_status'].copy()
    num_scale = [c for c in X.columns if c not in ['education', 'self_employed']]
    scaler = StandardScaler()
    X.loc[:, num_scale] = scaler.fit_transform(X[num_scale])
    return X, y, feature_cols, scaler, encoders


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

# ── PAGE 2: EDA ──────────────────────────────────────────────────────────────
elif page == "🔍 Exploratory Data Analysis":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🔍 Exploratory Data Analysis</h1>
        <p class="hero-subtitle">Understand the data before building models</p>
    </div>
    """, unsafe_allow_html=True)

    data = load_dataset()
    if data is None:
        st.stop()

    num_cols = [c for c in data.columns if data[c].dtype != object and c != 'loan_id']
    cat_cols = [c for c in data.columns if data[c].dtype == object and c != 'loan_status']

    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🎯 Feature vs Loan Status", "🔗 Correlations"])

    with tab1:
        st.markdown("#### Numerical Feature Distributions")
        sel_num = st.selectbox("Select numerical feature", num_cols, key="eda_num")
        fig = px.histogram(data, x=sel_num, color='loan_status', barmode='overlay',
                           template='plotly_dark',
                           title=f"Distribution of {sel_num} by Loan Status",
                           color_discrete_map={'Approved': '#F5A623', 'Rejected': '#4A90E2'})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Categorical Feature Distributions")
        for cat in cat_cols:
            vc = data[cat].value_counts().reset_index()
            fig_cat = px.bar(vc, x=cat, y='count', template='plotly_dark',
                             title=f"{cat} Distribution", color=cat)
            fig_cat.update_layout(showlegend=False, xaxis_title=cat, yaxis_title='Count')
            st.plotly_chart(fig_cat, use_container_width=True)

    with tab2:
        st.markdown("#### Numerical Feature vs Loan Status (Box Plot)")
        sel_box = st.selectbox("Select feature", num_cols, key="eda_box")
        fig_box = px.box(data, x='loan_status', y=sel_box, color='loan_status',
                         template='plotly_dark',
                         title=f"{sel_box} by Loan Status",
                         color_discrete_map={'Approved': '#F5A623', 'Rejected': '#4A90E2'})
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("#### Categorical Features vs Loan Status")
        for cat in cat_cols:
            ct = data.groupby([cat, 'loan_status']).size().reset_index(name='count')
            fig_ct = px.bar(ct, x=cat, y='count', color='loan_status', barmode='group',
                            template='plotly_dark',
                            title=f"{cat} by Loan Status",
                            color_discrete_map={'Approved': '#F5A623', 'Rejected': '#4A90E2'})
            st.plotly_chart(fig_ct, use_container_width=True)

    with tab3:
        st.markdown("#### Feature Correlation Heatmap")
        corr_df = data.drop(columns=['loan_id'], errors='ignore').copy()
        for col in corr_df.select_dtypes(include='object').columns:
            corr_df[col] = LabelEncoder().fit_transform(corr_df[col])
        corr_matrix = corr_df.corr()
        fig_heat = px.imshow(corr_matrix, template='plotly_dark',
                             title="Correlation Matrix",
                             color_continuous_scale='RdBu_r', aspect='auto',
                             text_auto='.2f')
        fig_heat.update_layout(height=600)
        st.plotly_chart(fig_heat, use_container_width=True)

        if 'loan_status' in corr_matrix.columns:
            st.markdown("#### Feature Correlations with Loan Status")
            top_c = (corr_matrix['loan_status']
                     .drop('loan_status')
                     .abs()
                     .sort_values(ascending=False)
                     .rename_axis('feature')
                     .reset_index(name='|correlation|'))
            fig_tc = px.bar(top_c, x='feature', y='|correlation|',
                            template='plotly_dark',
                            title="Absolute Correlation with Loan Status",
                            color='|correlation|',
                            color_continuous_scale='Oranges')
            fig_tc.update_layout(showlegend=False)
            st.plotly_chart(fig_tc, use_container_width=True)

# ── PAGE 3: DATA PREPROCESSING ───────────────────────────────────────────────
elif page == "⚙️ Data Preprocessing":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">⚙️ Data Preprocessing</h1>
        <p class="hero-subtitle">Cleaning, encoding and scaling the dataset</p>
    </div>
    """, unsafe_allow_html=True)

    data = load_dataset()
    if data is None:
        st.stop()

    st.markdown("### 1️⃣ Raw Data Overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rows", f"{data.shape[0]:,}")
    c2.metric("Total Columns", data.shape[1])
    c3.metric("Missing Values", int(data.isnull().sum().sum()))
    st.dataframe(data.head(5), use_container_width=True)

    st.markdown("### 2️⃣ Missing Value Analysis")
    missing = data.isnull().sum()
    if missing.sum() == 0:
        st.success("✅ No missing values found in the dataset!")
    else:
        st.dataframe(missing[missing > 0].rename('Missing Count').reset_index(),
                     use_container_width=True)

    st.markdown("### 3️⃣ Categorical Encoding (LabelEncoder)")
    st.markdown("""
| Column | Encoding |
|--------|----------|
| `education` | `Graduate → 0`,  `Not Graduate → 1` |
| `self_employed` | `No → 0`,  `Yes → 1` |
| `loan_status` | `Approved → 0`,  `Rejected → 1` |
    """)
    cat_cols_pp = [c for c in data.columns if data[c].dtype == object and c != 'loan_status']
    if cat_cols_pp:
        cols_pie = st.columns(len(cat_cols_pp))
        for i, cat in enumerate(cat_cols_pp):
            vc = data[cat].value_counts().reset_index()
            fig_pie = px.pie(vc, names=cat, values='count',
                             title=cat, template='plotly_dark')
            with cols_pie[i]:
                st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### 4️⃣ Feature Scaling (StandardScaler)")
    st.info("All numerical features are standardized to mean = 0, std = 1 before training.")
    num_demo = 'income_annum'
    if num_demo in data.columns:
        c_l, c_r = st.columns(2)
        with c_l:
            fig_b = px.histogram(data, x=num_demo, template='plotly_dark',
                                 title='Before Scaling', nbins=40,
                                 color_discrete_sequence=['#F5A623'])
            st.plotly_chart(fig_b, use_container_width=True)
        with c_r:
            scaled = (data[num_demo] - data[num_demo].mean()) / data[num_demo].std()
            fig_a = px.histogram(x=scaled, template='plotly_dark',
                                 title='After Scaling', nbins=40,
                                 color_discrete_sequence=['#4A90E2'])
            fig_a.update_layout(xaxis_title=num_demo)
            st.plotly_chart(fig_a, use_container_width=True)

    st.markdown("### 5️⃣ Train / Test Split (80 / 20)")
    n_test = int(len(data) * 0.2)
    n_train = len(data) - n_test
    fig_split = px.pie(values=[n_train, n_test], names=['Train', 'Test'],
                       title=f'80 / 20 Split  ·  Train: {n_train:,}  ·  Test: {n_test:,}',
                       template='plotly_dark',
                       color_discrete_map={'Train': '#F5A623', 'Test': '#4A90E2'})
    st.plotly_chart(fig_split, use_container_width=True)

    st.markdown("### 6️⃣ Preprocessed Data Sample")
    X_prev, y_prev, _, _, _ = preprocess_data(data)
    preview = X_prev.copy()
    preview['loan_status'] = y_prev.values
    st.dataframe(preview.head(10), use_container_width=True)

# ── PAGE 4: MODEL TRAINING ───────────────────────────────────────────────────
elif page == "🤖 Model Training":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🤖 Model Training</h1>
        <p class="hero-subtitle">Train Logistic Regression and Decision Tree classifiers</p>
    </div>
    """, unsafe_allow_html=True)

    data = load_dataset()
    if data is None:
        st.stop()

    col_cfg, col_res = st.columns([1, 3])
    with col_cfg:
        st.markdown("### ⚙️ Configuration")
        test_size = st.slider("Test Size", 0.10, 0.40, 0.20, 0.05)
        random_state = st.number_input("Random Seed", min_value=0, max_value=999, value=42)
        st.markdown("**Models:**")
        use_lr = st.checkbox("Logistic Regression", value=True)
        use_dt = st.checkbox("Decision Tree", value=True)
        train_btn = st.button("🚀 Train Models", use_container_width=True, type="primary")

    with col_res:
        if train_btn:
            if not use_lr and not use_dt:
                st.warning("Select at least one model.")
            else:
                with st.spinner("Training models..."):
                    X, y, feature_cols, scaler, encoders = preprocess_data(data)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=float(test_size),
                        random_state=int(random_state), stratify=y)
                    trained_models = {}
                    training_results = {}

                    if use_lr:
                        lr = LogisticRegression(max_iter=1000, random_state=int(random_state))
                        lr.fit(X_train, y_train)
                        yp = lr.predict(X_test)
                        yprob = lr.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, yprob)
                        training_results['Logistic Regression'] = {
                            'Accuracy': accuracy_score(y_test, yp),
                            'Precision': precision_score(y_test, yp, zero_division=0),
                            'Recall': recall_score(y_test, yp, zero_division=0),
                            'F1 Score': f1_score(y_test, yp, zero_division=0),
                            'ROC-AUC': roc_auc_score(y_test, yprob),
                            'cm': confusion_matrix(y_test, yp),
                            'fpr': fpr, 'tpr': tpr,
                            'importance': np.abs(lr.coef_[0]),
                        }
                        trained_models['Logistic Regression'] = lr

                    if use_dt:
                        dt = DecisionTreeClassifier(max_depth=10, random_state=int(random_state))
                        dt.fit(X_train, y_train)
                        yp = dt.predict(X_test)
                        yprob = dt.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, yprob)
                        training_results['Decision Tree'] = {
                            'Accuracy': accuracy_score(y_test, yp),
                            'Precision': precision_score(y_test, yp, zero_division=0),
                            'Recall': recall_score(y_test, yp, zero_division=0),
                            'F1 Score': f1_score(y_test, yp, zero_division=0),
                            'ROC-AUC': roc_auc_score(y_test, yprob),
                            'cm': confusion_matrix(y_test, yp),
                            'fpr': fpr, 'tpr': tpr,
                            'importance': dt.feature_importances_,
                        }
                        trained_models['Decision Tree'] = dt

                    st.session_state.trained_models = trained_models
                    st.session_state.training_results = training_results
                    st.session_state.feature_cols = feature_cols
                    st.session_state.scaler = scaler
                    st.session_state.encoders = encoders

                st.success("✅ Training complete! Results below.")

    if 'training_results' in st.session_state and st.session_state.training_results:
        results = st.session_state.training_results
        feature_cols = st.session_state.feature_cols
        metrics_keys = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']

        for model_name, res in results.items():
            st.markdown(f"---\n### 📈 {model_name}")
            m_cols = st.columns(5)
            for i, mk in enumerate(metrics_keys):
                m_cols[i].metric(mk, f"{res[mk]:.4f}")

            cm_col, imp_col = st.columns(2)
            with cm_col:
                cm = res['cm']
                fig_cm = px.imshow(cm, text_auto=True, template='plotly_dark',
                                   labels=dict(x='Predicted', y='Actual'),
                                   x=['Approved', 'Rejected'],
                                   y=['Approved', 'Rejected'],
                                   color_continuous_scale='Oranges',
                                   title='Confusion Matrix')
                st.plotly_chart(fig_cm, use_container_width=True)

            with imp_col:
                imp_df = (pd.DataFrame({'Feature': feature_cols,
                                        'Importance': res['importance']})
                          .sort_values('Importance', ascending=False).head(10))
                fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                                 template='plotly_dark',
                                 title='Top 10 Feature Importances',
                                 color='Importance',
                                 color_continuous_scale='Oranges')
                fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_imp, use_container_width=True)

# ── PAGE 5: MODEL COMPARISON ─────────────────────────────────────────────────
elif page == "⚖️ Model Comparison":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">⚖️ Model Comparison</h1>
        <p class="hero-subtitle">Side-by-side performance analysis</p>
    </div>
    """, unsafe_allow_html=True)

    if 'training_results' not in st.session_state or not st.session_state.training_results:
        st.warning("⚠️ No trained models found. Go to **Model Training** and train models first.")
        st.stop()

    results = st.session_state.training_results
    feature_cols = st.session_state.feature_cols
    metrics_keys = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    palette = ['#F5A623', '#4A90E2', '#7ED321', '#BD10E0']

    st.markdown("### 📊 Metrics Comparison")
    fig_bar = go.Figure()
    for i, (model_name, res) in enumerate(results.items()):
        fig_bar.add_trace(go.Bar(
            name=model_name,
            x=metrics_keys,
            y=[res[mk] for mk in metrics_keys],
            marker_color=palette[i % len(palette)]
        ))
    fig_bar.update_layout(barmode='group', template='plotly_dark',
                          title='Model Performance Metrics',
                          yaxis=dict(range=[0, 1], title='Score'),
                          xaxis_title='Metric')
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 📋 Metrics Summary Table")
    table_rows = {model_name: {mk: round(res[mk], 4) for mk in metrics_keys}
                  for model_name, res in results.items()}
    st.dataframe(pd.DataFrame(table_rows).T, use_container_width=True)

    st.markdown("### 📈 ROC Curves")
    fig_roc = go.Figure()
    for i, (model_name, res) in enumerate(results.items()):
        fig_roc.add_trace(go.Scatter(
            x=res['fpr'], y=res['tpr'], mode='lines',
            name=f"{model_name}  (AUC = {res['ROC-AUC']:.3f})",
            line=dict(color=palette[i % len(palette)], width=2)
        ))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  name='Random Classifier',
                                  line=dict(dash='dash', color='gray')))
    fig_roc.update_layout(template='plotly_dark', title='ROC Curves',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate',
                          xaxis=dict(range=[0, 1]),
                          yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("### 🔍 Feature Importance Comparison (Top 10)")
    fig_feat = go.Figure()
    for i, (model_name, res) in enumerate(results.items()):
        imp_df = (pd.DataFrame({'Feature': feature_cols, 'Importance': res['importance']})
                  .sort_values('Importance', ascending=False).head(10))
        fig_feat.add_trace(go.Bar(
            name=model_name,
            x=imp_df['Feature'],
            y=imp_df['Importance'],
            marker_color=palette[i % len(palette)]
        ))
    fig_feat.update_layout(barmode='group', template='plotly_dark',
                           title='Feature Importance by Model',
                           xaxis_title='Feature', yaxis_title='Importance')
    st.plotly_chart(fig_feat, use_container_width=True)

# ── PAGE 6: LIVE PREDICTOR ───────────────────────────────────────────────────
elif page == "🎯 Live Predictor":
    st.markdown("""
    <div class="hero-section">
        <h1 class="hero-title">🎯 Live Loan Approval Predictor</h1>
        <p class="hero-subtitle">Enter applicant details and get an instant prediction</p>
    </div>
    """, unsafe_allow_html=True)

    if 'trained_models' not in st.session_state or not st.session_state.trained_models:
        st.warning("⚠️ No trained models found. Go to **Model Training** and train models first.")
        st.stop()

    model_options = list(st.session_state.trained_models.keys())
    sel_model_name = st.selectbox("🤖 Select Model", model_options)

    st.markdown("### 📝 Applicant Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Personal Info**")
        no_of_dependents = st.slider("No. of Dependents", 0, 5, 2)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        income_annum = st.number_input("Annual Income (₹)", min_value=200000,
                                        max_value=10000000, value=5000000, step=100000)
    with c2:
        st.markdown("**Loan Details**")
        loan_amount = st.number_input("Loan Amount (₹)", min_value=300000,
                                       max_value=40000000, value=5000000, step=100000)
        loan_term = st.slider("Loan Term (years)", 2, 20, 10)
        cibil_score = st.slider("CIBIL Score", 300, 900, 600)

    with c3:
        st.markdown("**Asset Values**")
        residential_assets_value = st.number_input("Residential Assets (₹)",
                                                    min_value=0, max_value=30000000,
                                                    value=5000000, step=100000)
        commercial_assets_value = st.number_input("Commercial Assets (₹)",
                                                   min_value=0, max_value=30000000,
                                                   value=2000000, step=100000)
        luxury_assets_value = st.number_input("Luxury Assets (₹)",
                                               min_value=0, max_value=40000000,
                                               value=5000000, step=100000)
        bank_asset_value = st.number_input("Bank Assets (₹)",
                                            min_value=0, max_value=15000000,
                                            value=3000000, step=100000)

    st.markdown("---")
    if st.button("🎯 Predict Loan Approval", use_container_width=True, type="primary"):
        model = st.session_state.trained_models[sel_model_name]
        encoders = st.session_state.encoders
        scaler = st.session_state.scaler
        feature_cols = st.session_state.feature_cols

        raw_input = {
            'no_of_dependents': no_of_dependents,
            'education': int(encoders['education'].transform([education])[0]),
            'self_employed': int(encoders['self_employed'].transform([self_employed])[0]),
            'income_annum': income_annum,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'cibil_score': cibil_score,
            'residential_assets_value': residential_assets_value,
            'commercial_assets_value': commercial_assets_value,
            'luxury_assets_value': luxury_assets_value,
            'bank_asset_value': bank_asset_value,
        }
        input_df = pd.DataFrame([raw_input])[feature_cols].copy()
        num_scale = [c for c in feature_cols if c not in ['education', 'self_employed']]
        input_df.loc[:, num_scale] = scaler.transform(input_df[num_scale])

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        le_target = encoders['loan_status']
        result_label = le_target.inverse_transform([pred])[0]
        class_labels = list(le_target.classes_)

        if result_label == 'Approved':
            conf = proba[class_labels.index('Approved')] * 100
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#0d2b0d,#1a5c1a);
    padding:2rem;border-radius:15px;border:2px solid #4CAF50;
    text-align:center;margin:1rem 0;">
  <h2 style="color:#4CAF50;margin:0;font-size:2rem">✅ LOAN APPROVED</h2>
  <p style="color:#a0e8a0;font-size:1.2rem;margin-top:0.5rem">
    Confidence: <strong>{conf:.1f}%</strong>
  </p>
</div>""", unsafe_allow_html=True)
        else:
            conf = proba[class_labels.index('Rejected')] * 100
            st.markdown(f"""
<div style="background:linear-gradient(135deg,#2b0d0d,#5c1a1a);
    padding:2rem;border-radius:15px;border:2px solid #F44336;
    text-align:center;margin:1rem 0;">
  <h2 style="color:#F44336;margin:0;font-size:2rem">❌ LOAN REJECTED</h2>
  <p style="color:#e8a0a0;font-size:1.2rem;margin-top:0.5rem">
    Confidence: <strong>{conf:.1f}%</strong>
  </p>
</div>""", unsafe_allow_html=True)

        st.markdown("### 📊 Prediction Probabilities")
        fig_prob = px.bar(
            x=class_labels, y=list(proba),
            template='plotly_dark',
            title='Class Probabilities',
            color=class_labels,
            color_discrete_map={'Approved': '#4CAF50', 'Rejected': '#F44336'},
            text=[f"{p*100:.1f}%" for p in proba]
        )
        fig_prob.update_layout(showlegend=False,
                               yaxis=dict(range=[0, 1], title='Probability'),
                               xaxis_title='Outcome')
        fig_prob.update_traces(textposition='outside')
        st.plotly_chart(fig_prob, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #B0B0B0; padding: 1rem;">
    <p>🏦 Loan Approval Prediction System | Built with Streamlit & Machine Learning</p>
    <p>© 2024 | Advanced Analytics for Financial Decision Making</p>
</div>
""", unsafe_allow_html=True)

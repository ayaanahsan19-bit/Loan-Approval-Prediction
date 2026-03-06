# 🏦 Loan Approval Prediction System

An interactive web application for intelligent loan approval prediction using Machine Learning, developed by **Ayaan Ahsan**. This project demonstrates a full data science workflow — from exploratory analysis to live predictions — built with Streamlit and scikit-learn.

## 🌟 Features

### 📊 **6-Page Interactive Dashboard**
- **Home / Overview**: KPI metric cards, loan status distribution chart, dataset preview
- **Exploratory Data Analysis**: Distribution plots, box plots, correlation heatmap
- **Data Preprocessing**: Encoding, scaling and train/test split walkthrough
- **Model Training**: Configurable LR & Decision Tree training with metrics and confusion matrices
- **Model Comparison**: Side-by-side metrics, ROC curves, feature importance comparison
- **Live Predictor**: Real-time prediction with confidence scores and probability chart

### 🤖 **Machine Learning**
- **Two models**: Logistic Regression & Decision Tree Classifier
- **Full preprocessing pipeline**: Median/mode imputation → LabelEncoder → StandardScaler
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **In-session persistence**: Trained models and scalers stored in `st.session_state`

### 🎨 **UI / UX**
- Dark theme (`#0D0D0D`) with gold accents (`#F5A623`)
- Fully responsive Plotly charts (`plotly_dark` template)
- Hero sections and metric cards via custom inline CSS

## 🛠️ Technology Stack

| Package | Version |
|---|---|
| Python | 3.12 |
| Streamlit | 1.40.0 |
| Pandas | 2.2.1 |
| NumPy | 1.26.4 |
| scikit-learn | 1.4.2 |
| Plotly | 5.20.0 |
| PyArrow | 17.0.0 |

## 📁 Project Structure

```
loan-approval-prediction/
├── app.py                      # Main Streamlit application (all 6 pages)
├── loan_approval_dataset.csv   # Dataset
├── requirements.txt            # Pinned dependencies
├── .python-version             # Pins Python 3.12 for Streamlit Cloud
├── src/
│   ├── preprocess.py           # Preprocessing utilities
│   ├── model.py                # Model training & evaluation classes
│   └── visualizations.py      # Chart helper functions
├── assets/
│   └── style.css               # Additional CSS
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ayaanahsan19-bit/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
```

2. **Create virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

App opens automatically at `http://localhost:8501`

### Dataset
The dataset (`loan_approval_dataset.csv`) is included in the repository.  
Expected columns:
```
loan_id, no_of_dependents, education, self_employed, income_annum,
loan_amount, loan_term, cibil_score, residential_assets_value,
commercial_assets_value, luxury_assets_value, bank_asset_value, loan_status
```

## 📊 Application Pages

### 🏠 Home / Overview
- Total applications, feature count, approval rate, missing value count
- Loan status distribution bar chart
- Dataset preview and column type breakdown

### 🔍 Exploratory Data Analysis
- **Distributions tab**: Histogram per numerical feature split by loan status; bar charts for categoricals
- **Feature vs Loan Status tab**: Box plots + grouped bar charts per categorical
- **Correlations tab**: Full correlation heatmap + ranked absolute-correlation bar chart

### ⚙️ Data Preprocessing
- Raw data shape, missing value audit
- Encoding reference table (education / self_employed / loan_status)
- Before/after scaling histograms (StandardScaler)
- 80/20 train-test split pie chart
- Preprocessed data sample

### 🤖 Model Training
- Configure test size (10–40%), random seed, and which models to train
- Trains selected models and displays: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion matrix heatmap and top-10 feature importance bar per model
- Results stored in session state for use in Comparison and Live Predictor

### ⚖️ Model Comparison
- Grouped bar chart of all 5 metrics
- Metrics summary table
- Overlaid ROC curves with AUC labels
- Side-by-side top-10 feature importance

### 🎯 Live Predictor
- 11-field input form (sliders + number inputs)
- Applies the same encoding + scaling pipeline used during training
- Green **APPROVED** / Red **REJECTED** banner with confidence percentage
- Probability bar chart for both outcome classes

> **Note**: Model Comparison and Live Predictor require models to be trained first on the Model Training page.

## 🌐 Deployment

### Streamlit Cloud
The repo includes `.python-version` (pinned to `3.12`) and all dependencies have pre-built Python 3.12 wheels — no source compilation occurs on deployment.

1. Push to GitHub
2. Connect repo at [share.streamlit.io](https://share.streamlit.io)
3. Set main file to `app.py`
4. Deploy

### Docker
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📈 Future Enhancements

- [ ] **Additional Models**: Random Forest, XGBoost, Neural Networks
- [ ] **API Integration**: Real-time credit bureau data
- [ ] **Advanced Analytics**: Customer segmentation, risk scoring
- [ ] **Multi-language Support**: International deployment
- [ ] **Mobile App**: React Native companion application

## �‍💻 About the Developer

**Ayaan Ahsan** is a passionate Machine Learning engineer and Python developer with expertise in building end-to-end data science solutions. This project demonstrates his proficiency in:

- **Machine Learning**: Implementing and comparing multiple ML algorithms
- **Data Science**: Complete data preprocessing and analysis pipelines
- **Web Development**: Creating interactive applications with Streamlit
- **Visualization**: Building professional dashboards with Plotly
- **Software Engineering**: Following best practices with modular code architecture

Ayaan combines technical expertise with a keen eye for user experience, creating applications that are both powerful and intuitive.

## � Contact & Support

- **LinkedIn**: [www.linkedin.com/in/ayaan-ahsan-39bb422bb](https://www.linkedin.com/in/ayaan-ahsan-39bb422bb)
- **GitHub**: [github.com/ayaanahsan19-bit](https://github.com/ayaanahsan19-bit)
- **Email**: [ayaan.ahsan19@gmail.com](mailto:ayaan.ahsan19@gmail.com)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit Team**: For the amazing web app framework
- **Scikit-learn**: For comprehensive ML tools
- **Plotly**: For interactive visualizations
- **Data Science Community**: For inspiration and best practices

---

**⭐ If you find this project helpful, please give it a star on GitHub!**

**🚀 This project demonstrates advanced ML engineering skills and is ready to transform loan approval processes with AI. Connect with Ayaan to discuss how this technology can revolutionize financial decision-making!**

---

#MachineLearning #DataScience #FinTech #Streamlit #Python #LoanApproval #ArtificialIntelligence #MLOps #WebDevelopment

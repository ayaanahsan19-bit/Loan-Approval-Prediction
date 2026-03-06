"""
Model training and evaluation module for loan approval prediction.
Handles ML model training, evaluation, and comparison.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from imblearn.over_sampling import SMOTE
import joblib
from typing import Tuple, Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px


class LoanApprovalModel:
    """
    A class to handle loan approval prediction models.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.results = {}
    
    def train_logistic_regression(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                random_state: int = 42) -> LogisticRegression:
        """
        Train Logistic Regression model.
        
        Args:
            X_train: Training features
            y_train: Training target
            random_state: Random seed for reproducibility
            
        Returns:
            Trained Logistic Regression model
        """
        lr = LogisticRegression(random_state=random_state, max_iter=1000)
        lr.fit(X_train, y_train)
        self.models['logistic_regression'] = lr
        return lr
    
    def train_decision_tree(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          random_state: int = 42) -> DecisionTreeClassifier:
        """
        Train Decision Tree model.
        
        Args:
            X_train: Training features
            y_train: Training target
            random_state: Random seed for reproducibility
            
        Returns:
            Trained Decision Tree model
        """
        dt = DecisionTreeClassifier(random_state=random_state, max_depth=10)
        dt.fit(X_train, y_train)
        self.models['decision_tree'] = dt
        return dt
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to handle class imbalance.
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Resampled features and target
        """
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for results storage
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.results[model_name] = metrics
        return metrics
    
    def train_and_evaluate_models(self, X: pd.DataFrame, y: pd.Series, 
                                test_size: float = 0.2, random_state: int = 42,
                                apply_smote_flag: bool = False) -> Dict[str, Any]:
        """
        Train and evaluate both models with optional SMOTE.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion of data for testing
            random_state: Random seed
            apply_smote_flag: Whether to apply SMOTE
            
        Returns:
            Dictionary with all results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Apply SMOTE if requested
        if apply_smote_flag:
            X_train, y_train = self.apply_smote(X_train, y_train)
        
        # Train models
        lr_model = self.train_logistic_regression(X_train, y_train, random_state)
        dt_model = self.train_decision_tree(X_train, y_train, random_state)
        
        # Evaluate models
        lr_results = self.evaluate_model(lr_model, X_test, y_test, 'logistic_regression')
        dt_results = self.evaluate_model(dt_model, X_test, y_test, 'decision_tree')
        
        return {
            'train_test_split': {'X_train': X_train, 'X_test': X_test, 
                               'y_train': y_train, 'y_test': y_test},
            'logistic_regression': lr_results,
            'decision_tree': dt_results,
            'smote_applied': apply_smote_flag
        }
    
    def get_feature_importance(self, model_name: str, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model_name: Name of the model ('logistic_regression' or 'decision_tree')
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name == 'logistic_regression':
            importance = np.abs(model.coef_[0])
        else:  # decision_tree
            importance = model.feature_importances_
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def predict_single(self, model_name: str, features: pd.DataFrame) -> Tuple[int, float]:
        """
        Make prediction for a single applicant.
        
        Args:
            model_name: Name of the model to use
            features: Features for single prediction
            
        Returns:
            Tuple of (prediction, probability)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]
        
        return prediction, probability
    
    def save_models(self, filepath_prefix: str = "models/loan_model"):
        """
        Save trained models to disk.
        
        Args:
            filepath_prefix: Prefix for model file paths
        """
        import os
        os.makedirs('models', exist_ok=True)
        
        for model_name, model in self.models.items():
            joblib.dump(model, f"{filepath_prefix}_{model_name}.joblib")
        
        if self.scalers:
            joblib.dump(self.scalers, f"{filepath_prefix}_scalers.joblib")
        
        if self.encoders:
            joblib.dump(self.encoders, f"{filepath_prefix}_encoders.joblib")
    
    def load_models(self, filepath_prefix: str = "models/loan_model"):
        """
        Load trained models from disk.
        
        Args:
            filepath_prefix: Prefix for model file paths
        """
        try:
            self.models['logistic_regression'] = joblib.load(f"{filepath_prefix}_logistic_regression.joblib")
            self.models['decision_tree'] = joblib.load(f"{filepath_prefix}_decision_tree.joblib")
            self.scalers = joblib.load(f"{filepath_prefix}_scalers.joblib")
            self.encoders = joblib.load(f"{filepath_prefix}_encoders.joblib")
        except FileNotFoundError:
            print("No saved models found. Please train models first.")
    
    def create_confusion_matrix_plot(self, model_name: str) -> go.Figure:
        """
        Create confusion matrix plot using Plotly.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for model {model_name}")
        
        cm = self.results[model_name]['confusion_matrix']
        
        fig = px.imshow(cm, 
                       text_auto=True, 
                       color_continuous_scale='Blues',
                       title=f'Confusion Matrix - {model_name.replace("_", " ").title()}',
                       labels=dict(x="Predicted", y="Actual", color="Count"))
        
        return fig
    
    def create_roc_curve_plot(self, model_name: str) -> go.Figure:
        """
        Create ROC curve plot using Plotly.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for model {model_name}")
        
        # Get ROC curve data
        y_test = self.results[model_name].get('y_test')
        y_pred_proba = self.results[model_name]['probabilities']
        
        if y_test is None:
            raise ValueError("Test labels not available for ROC curve")
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = self.results[model_name]['roc_auc']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='gold', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name.replace("_", " ").title()}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            template='plotly_dark'
        )
        
        return fig

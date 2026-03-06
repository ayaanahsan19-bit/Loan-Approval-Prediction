"""
Data preprocessing module for loan approval prediction.
Handles cleaning, imputation, and encoding of the loan dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Any


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the loan approval dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Raw pandas DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at {file_path}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and duplicates.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Remove duplicates if any
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    removed_duplicates = initial_rows - len(df_clean)
    
    # Handle missing values for numerical columns with median
    numerical_cols = ['income_annum', 'loan_amount', 'cibil_score', 
                      'residential_assets_value', 'commercial_assets_value', 
                      'luxury_assets_value', 'bank_asset_value']
    
    for col in numerical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Handle missing values for categorical columns with mode
    categorical_cols = ['education', 'self_employed']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    
    # Convert loan_term to numeric if needed
    if 'loan_term' in df_clean.columns:
        df_clean['loan_term'] = pd.to_numeric(df_clean['loan_term'], errors='coerce')
        df_clean['loan_term'] = df_clean['loan_term'].fillna(df_clean['loan_term'].median())
    
    return df_clean


def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical features using label encoding.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        Tuple of (encoded DataFrame, dictionary of encoders)
    """
    df_encoded = df.copy()
    encoders = {}
    
    # Encode education
    if 'education' in df_encoded.columns:
        le_education = LabelEncoder()
        df_encoded['education'] = le_education.fit_transform(df_encoded['education'])
        encoders['education'] = le_education
    
    # Encode self_employed
    if 'self_employed' in df_encoded.columns:
        le_self_employed = LabelEncoder()
        df_encoded['self_employed'] = le_self_employed.fit_transform(df_encoded['self_employed'])
        encoders['self_employed'] = le_self_employed
    
    # Encode target variable (loan_status)
    if 'loan_status' in df_encoded.columns:
        le_loan_status = LabelEncoder()
        df_encoded['loan_status'] = le_loan_status.fit_transform(df_encoded['loan_status'])
        encoders['loan_status'] = le_loan_status
    
    return df_encoded, encoders


def scale_features(df: pd.DataFrame, target_col: str = 'loan_status') -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df: Encoded DataFrame
        target_col: Name of target column to exclude from scaling
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    df_scaled = df.copy()
    
    # Identify numerical columns (exclude target and categorical encoded columns)
    numerical_cols = ['income_annum', 'loan_amount', 'cibil_score', 
                     'residential_assets_value', 'commercial_assets_value', 
                     'luxury_assets_value', 'bank_asset_value', 'loan_term']
    
    numerical_cols = [col for col in numerical_cols if col in df_scaled.columns and col != target_col]
    
    if numerical_cols:
        scaler = StandardScaler()
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
    else:
        scaler = StandardScaler()
    
    return df_scaled, scaler


def prepare_features_and_target(df: pd.DataFrame, target_col: str = 'loan_status') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable.
    
    Args:
        df: Preprocessed DataFrame
        target_col: Name of target column
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Drop loan_id as it's not useful for prediction
    feature_cols = [col for col in df.columns if col not in [target_col, 'loan_id']]
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y


def get_preprocessing_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get information about the preprocessing steps.
    
    Args:
        df: Original DataFrame
        
    Returns:
        Dictionary with preprocessing information
    """
    info = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'target_distribution': df['loan_status'].value_counts().to_dict() if 'loan_status' in df.columns else {},
        'approval_rate': (df['loan_status'].value_counts().get('Approved', 0) / len(df) * 100) if 'loan_status' in df.columns else 0
    }
    
    return info

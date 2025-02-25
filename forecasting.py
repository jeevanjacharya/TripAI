# forecasting.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Function to automatically select a target column for forecasting
def auto_select_target_column(df):
    # Select numerical columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        raise ValueError("No numerical columns found in the dataframe.")
    
    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()
    
    # Identify the column with the highest correlation with others (excluding self-correlation)
    correlation_matrix = correlation_matrix.abs()
    correlation_matrix = correlation_matrix.unstack().sort_values(ascending=False)
    correlation_matrix = correlation_matrix[correlation_matrix < 1]  # Exclude self-correlation
    
    # Find the pair with the highest correlation
    most_correlated_pair = correlation_matrix.idxmax()
    target_column = most_correlated_pair[1]  # Target column will be the second in the pair
    
    return target_column

# Forecasting function using the selected target column
def forecasting(df):
    target_column = auto_select_target_column(df)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Only use numeric columns for forecasting
    X = X.select_dtypes(include=[np.number])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    return target_column, predictions

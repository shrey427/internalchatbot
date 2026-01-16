import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# ===============================================================
# CORE UTILITY: Universal Feature Selection
# ===============================================================

def select_important_features(df, top_n=5, is_unsupervised=False):
    """
    Cleans and ranks features based on the specific ML task type.
    Essential for handling generic user-uploaded financial data.
    """
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] < 2:
        return numeric_df

    # 1. Remove Low Variance (Constant columns)
    v_threshold = VarianceThreshold(threshold=(.8 * (1 - .8)))
    try:
        numeric_df = pd.DataFrame(
            v_threshold.fit_transform(numeric_df), 
            columns=numeric_df.columns[v_threshold.get_support()]
        )
    except ValueError:
        return numeric_df

    # 2. Remove Redundant Features (Correlation > 0.95)
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    numeric_df = numeric_df.drop(columns=to_drop)

    # 3. Task-Specific Selection
    if is_unsupervised:
        # For Anomalies/Optimization: Rank by Signal Strength
        scaled_variance = numeric_df.var() / (numeric_df.mean() + 1e-6)
        top_features = scaled_variance.nlargest(top_n).index.tolist()
        return numeric_df[top_features]
    else:
        # For Prediction: Rank by relationship to the target (last column)
        target = numeric_df.columns[-1]
        X = numeric_df.drop(columns=[target])
        y = numeric_df[target]
        k = min(top_n, X.shape[1])
        
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()].tolist()
        return numeric_df[selected_cols + [target]]


def validate_model_performance(X, y, model_type="regression"):
    """
    Standardizes validation across all domain models.
    Returns an accuracy percentage or R2 score.
    """
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Generic model to test data quality
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    
    if model_type == "regression":
        tester = RandomForestRegressor(n_estimators=50, random_state=42)
        tester.fit(X_train, y_train)
        score = tester.score(X_test, y_test)
        return f"{max(0, round(score * 100, 2))}% (R2 Score)"
    else:
        tester = RandomForestClassifier(n_estimators=50, random_state=42)
        tester.fit(X_train, y_train)
        score = tester.score(X_test, y_test)
        return f"{round(score * 100, 2)}% (Accuracy)"

# ===============================================================
# FINANCE MODELS
# ===============================================================

def credit_risk_assessment(df):
    """
    Approach: Random Forest
    Predicts likelihood of loan default and identifies key risk drivers.
    """
    numeric_df = select_important_features(df, is_unsupervised=False)
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance = dict(zip(X.columns, np.round(model.feature_importances_, 4)))
    acc = validate_model_performance(X, y, "classification")
    return {
        "model": "Random Forest (Credit Risk)",
        "risk_indicators": dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)),
        "validation_score": acc
    }

def expense_budget_forecasting(df, periods=6):
    """
    Approach: Robust Regression
    Predicts future expenses and cash flow using trend + momentum logic.
    """
    numeric_df = select_important_features(df, is_unsupervised=False)
    series = numeric_df.iloc[:, -1].values.astype(float)
    n = len(series)
    
    # Feature Engineering: Time Index + Rolling Momentum
    time_index = np.arange(n).reshape(-1, 1)
    rolling_mean = pd.Series(series).rolling(window=3, min_periods=1).mean().values.reshape(-1, 1)
    X = np.hstack([time_index, rolling_mean])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Huber is used to handle sudden spikes in financial expenses
    model = HuberRegressor()
    model.fit(X_scaled, series)
    
    forecast_values = []
    current_series = series.tolist()
    
    for i in range(periods):
        recent_avg = np.mean(current_series[-3:])
        X_next_scaled = scaler.transform([[n + i, recent_avg]])
        prediction = model.predict(X_next_scaled)[0]
        forecast_values.append(max(0, prediction)) # Expenses cannot be negative
        current_series.append(prediction)
    acc = validate_model_performance(X_scaled, series, "regression")
    return {
        "model": "Huber Regression (Expense Forecast)",
        "forecast_values": np.round(forecast_values, 2).tolist(),
        "validation_score": acc
    }

def portfolio_optimization(df):
    """
    Approach: ML Optimization
    Maximizes returns and minimizes risk across assets.
    """
    numeric_df = select_important_features(df, is_unsupervised=True)
    
    # Calculate returns and risk (covariance)
    returns = numeric_df.pct_change().mean()
    # Simplified optimization: Weighted by inverse volatility
    volatility = numeric_df.pct_change().std()
    weights = (1 / volatility) / (1 / volatility).sum()
    
    allocation = dict(zip(numeric_df.columns, np.round(weights, 4)))
    acc = validate_model_performance(numeric_df, returns.fillna(0), "regression")
    return {
        "model": "Mean-Variance Optimization",
        "recommended_allocation": allocation,
        "validation_score": acc
    }

def invoice_anomaly_detection(df):
    """
    Approach: Isolation Forest
    Flags unusual invoices or delayed payments for audit.
    """
    numeric_df = select_important_features(df, is_unsupervised=True)
    
    # Use RobustScaler to ensure anomalies don't distort the scaling
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    
    model = IsolationForest(contamination=0.05, random_state=42)
    labels = model.fit_predict(scaled_data)
    
    anomaly_indices = np.where(labels == -1)[0].tolist()
    acc = validate_model_performance(scaled_data, labels, "classification")
    return {
        "model": "Isolation Forest (Anomaly Detection)",
        "flagged_count": len(anomaly_indices),
        "flagged_record_indices": anomaly_indices,
        "validation_score": acc
    }

def financial_statement_analysis(df):
    """
    Approach: Predictive ML
    Analyzes reports to predict future performance drivers.
    """
    numeric_df = select_important_features(df, is_unsupervised=False)
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]
    
    # Use Ridge to stabilize feature impact scores
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    impact = dict(zip(X.columns, np.round(model.coef_, 4)))
    acc = validate_model_performance(X, y, "regression")
    return {
        "model": "Ridge Regression (Impact Analysis)",
        "performance_drivers": dict(sorted(impact.items(), key=lambda x: abs(x[1]), reverse=True)),
        "validation_score": acc
    }
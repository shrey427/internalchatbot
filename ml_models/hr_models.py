import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.metrics.pairwise import cosine_similarity

# ===============================================================
# CORE UTILITY: Universal Feature Selection
# ===============================================================

def select_important_features(df, top_n=5, is_unsupervised=False):
    """
    Cleans and ranks HR features based on the specific ML task.
    Crucial for identifying risk factors in Attrition or Workforce Planning.
    """
    numeric_df = df.select_dtypes(include=["number"]).copy()
    if numeric_df.shape[1] < 2:
        return numeric_df

    # 1. Remove Low Variance (Constant columns like 'Employee Status' if everyone is Active)
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
        # For Clustering/Training Recommendations: Rank by spread
        scaled_variance = numeric_df.var() / (numeric_df.mean() + 1e-6)
        top_features = scaled_variance.nlargest(top_n).index.tolist()
        return numeric_df[top_features]
    else:
        # For Attrition Prediction: Rank by relationship to Target
        target = numeric_df.columns[-1]
        X = numeric_df.drop(columns=[target])
        y = numeric_df[target]
        k = min(top_n, X.shape[1])
        
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_cols = X.columns[selector.get_support()].tolist()
        return numeric_df[selected_cols + [target]]

# ===============================================================
# HR MODELS
# ===============================================================

def attrition_turnover_prediction(df):
    """
    Approach: Classification (Random Forest/XGBoost logic)
    Identifies employees at risk of leaving to design retention strategies.
    """
    numeric_df = select_important_features(df, is_unsupervised=False)
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]
    
    # Random Forest is highly effective for tabular HR data classification
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    importance = dict(zip(X.columns, np.round(model.feature_importances_, 4)))
    
    return {
        "model": "Classification (Attrition Risk)",
        "turnover_drivers": dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)),
        "outcome": "Risk factors identified for retention planning."
    }

def workforce_planning(df, periods=6):
    """
    Approach: Forecasting / Clustering
    Optimizes staffing levels based on historical trends and demand.
    """
    numeric_df = select_important_features(df, is_unsupervised=False)
    series = numeric_df.iloc[:, -1].values.astype(float)
    
    # Robust Forecasting logic (Trend + Momentum)
    time_index = np.arange(len(series)).reshape(-1, 1)
    rolling_mean = pd.Series(series).rolling(window=3, min_periods=1).mean().values.reshape(-1, 1)
    X = np.hstack([time_index, rolling_mean])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = HuberRegressor().fit(X_scaled, series)
    
    forecast_values = []
    current_series = series.tolist()
    
    for i in range(periods):
        recent_avg = np.mean(current_series[-3:])
        X_next_scaled = scaler.transform([[len(current_series), recent_avg]])
        prediction = model.predict(X_next_scaled)[0]
        forecast_values.append(max(0, round(prediction))) # Staffing counts must be integers
        current_series.append(prediction)

    return {
        "model": "Forecasting (Staffing Demand)",
        "projected_staffing_needs": forecast_values,
        "value": "Optimized staffing levels based on trends."
    }

def training_recommendation(df, top_n=3):
    """
    Approach: Recommendation Systems
    Suggests personalized training programs based on employee profiles.
    """
    numeric_df = select_important_features(df, is_unsupervised=True)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    similarity = cosine_similarity(scaled_data)

    results = {}
    # Sample top 5 employees to show recommendations for
    for idx in range(min(5, len(similarity))):
        similar_indices = np.argsort(similarity[idx])[::-1][1:top_n + 1]
        results[f"Employee_{idx}"] = {
            "peer_group_matches": similar_indices.tolist(),
            "note": "Suggest courses completed by these peers."
        }

    return {
        "model": "Cosine Similarity Recommendation",
        "recommendations": results
    }

def salary_compensation_benchmarking(df):
    """
    Approach: Regression / Clustering
    Predicts fair salaries based on role, location, and experience.
    """
    numeric_df = select_important_features(df, is_unsupervised=False)
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]
    
    # Ridge regression helps maintain fair benchmarking without overfitting
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    benchmarks = dict(zip(X.columns, np.round(model.coef_, 4)))
    
    return {
        "model": "Regression (Fair Pay Benchmarking)",
        "compensation_drivers": benchmarks,
        "outcome": "Predict fair salaries based on experience and role."
    }
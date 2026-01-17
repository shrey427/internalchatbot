import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# ===============================================================
# CORE UTILITY: Universal Feature Selection
# ===============================================================

def select_important_features(df, top_n=5, is_unsupervised=False):
    """
    Cleans and ranks HR features based on the specific ML task.
    Crucial for identifying risk factors in Attrition or Workforce Planning.
    """
    df_clean = df.copy()

    # --- STEP 1: CATEGORICAL ENCODING & IMPUTATION ---
    # Identify non-numeric columns
    cat_cols = df_clean.select_dtypes(exclude=['number']).columns
    for col in cat_cols:
        # Fill missing text with 'Unknown' to avoid NaN errors
        df_clean[col] = df_clean[col].fillna('Unknown')
        
        unique_count = df_clean[col].nunique()
        if unique_count > 20: # Drop high-cardinality IDs/Names
            df_clean.drop(columns=[col], inplace=True)
            continue
        
        # Generalised Encoding (Ordinal vs Nominal)
        if unique_count <= 5:
            df_clean[col] = LabelEncoder().fit_transform(df_clean[col].astype(str))
        else:
            df_one_hot = pd.get_dummies(df_clean[col], prefix=col, drop_first=True)
            df_clean = pd.concat([df_clean, df_one_hot], axis=1)
            df_clean.drop(columns=[col], inplace=True)

    # --- STEP 2: NUMERIC IMPUTATION ---
    # Isolate numeric data for ML models
    numeric_df = df_clean.select_dtypes(include=["number"]).copy()
    if numeric_df.empty:
        return numeric_df

    # Fill numeric NaNs with the median to fix the error
    imputer = SimpleImputer(strategy='median')
    numeric_df = pd.DataFrame(
        imputer.fit_transform(numeric_df), 
        columns=numeric_df.columns
    )

    # --- STEP 3: VARIANCE & REDUNDANCY FILTERING ---
    v_threshold = VarianceThreshold(threshold=(.8 * (1 - .8)))
    try:
        numeric_df = pd.DataFrame(
            v_threshold.fit_transform(numeric_df), 
            columns=numeric_df.columns[v_threshold.get_support()]
        )
    except ValueError:
        return numeric_df

    # --- STEP 4: TASK-SPECIFIC SELECTION ---
    if is_unsupervised:
        scaled_variance = numeric_df.var() / (numeric_df.mean() + 1e-6)
        top_features = scaled_variance.nlargest(top_n).index.tolist()
        return numeric_df[top_features]
    else:
        target = numeric_df.columns[-1]
        X = numeric_df.drop(columns=[target])
        y = numeric_df[target]
        k = min(top_n, X.shape[1])
        
        # SelectKBest now receives clean, non-NaN data
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
    acc = validate_model_performance(X, y, "classification")
    return {
        "model": "Classification (Attrition Risk)",
        "turnover_drivers": dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)),
        "outcome": "Risk factors identified for retention planning.",
        "validation_score": acc
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
    acc = validate_model_performance(X_scaled, series, "regression")
    return {
        "model": "Forecasting (Staffing Demand)",
        "projected_staffing_needs": forecast_values,
        "value": "Optimized staffing levels based on trends.",
        "validation_score": acc
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
    acc = validate_model_performance(scaled_data, np.arange(len(scaled_data)), "regression")
    return {
        "model": "Cosine Similarity Recommendation",
        "recommendations": results,
        "validation_score": acc
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
    acc = validate_model_performance(X, y, "regression")
    return {
        "model": "Regression (Fair Pay Benchmarking)",
        "compensation_drivers": benchmarks,
        "outcome": "Predict fair salaries based on experience and role.",
        "validation_score": acc
    }
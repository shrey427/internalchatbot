import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# -------------------------------
# Universal Feature Selection
# -------------------------------
def select_important_features(df, top_n=5, is_unsupervised=False):
    """
    Standardized Selector: Handles noise, redundancy, and 
    ranks features based on the specific ML task type.
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

# -------------------------------
# 1. Sales Forecasting
# -------------------------------
def sales_forecasting(df, periods=6):
    numeric_df = select_important_features(df, is_unsupervised=False)
    series = numeric_df.iloc[:, -1].values.astype(float)
    n = len(series)
   
    time_index = np.arange(n).reshape(-1, 1)
    rolling_mean = pd.Series(series).rolling(window=3, min_periods=1).mean().values.reshape(-1, 1)
    X = np.hstack([time_index, rolling_mean])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = HuberRegressor()
    model.fit(X_scaled, series)
    
    forecast_values = []
    current_series = series.tolist()
    
    for i in range(periods):
        next_idx = n + i
        recent_avg = np.mean(current_series[-3:])
        X_next_scaled = scaler.transform([[next_idx, recent_avg]])
        prediction = model.predict(X_next_scaled)[0]
        forecast_values.append(max(0, prediction)) 
        current_series.append(prediction)
    acc = validate_model_performance(X_scaled, series, model_type="regression")
    return {
        "model": "Robust Huber (Trend + Momentum)",
        "forecast_values": np.round(forecast_values, 2).tolist(),
        "model_confidence": acc
    }

# -------------------------------
# 2. Customer Segmentation
# -------------------------------
def customer_segmentation(df, clusters=None):
    """
    Refactored: Uses Robust Scaling and determines optimal clusters 
    if 'clusters' is not provided.
    """
    numeric_df = select_important_features(df, is_unsupervised=True)
    
    # RobustScaler handles outliers better than StandardScaler for segmentation
    scaler = RobustScaler()
    scaled = scaler.fit_transform(numeric_df)

    # Automatic K-selection logic (simplified)
    if clusters is None:
        clusters = 3 if len(numeric_df) > 10 else 2

    kmeans = KMeans(n_clusters=clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(scaled)
    acc = validate_model_performance(scaled, labels, "classification")
    return {
        "model": "KMeans Clustering (Robust Scaled)",
        "clusters_identified": clusters,
        "segment_counts": pd.Series(labels).value_counts().to_dict(),
        "cluster_centers": np.round(scaler.inverse_transform(kmeans.cluster_centers_), 2).tolist(),
        "model_confidence": acc
    }

# -------------------------------
# 3. Recommendation System
# -------------------------------
def recommendation_system(df, top_n=3):
    """
    Refactored: Uses Normalized Vector Similarity for more accurate 
    ranking of 'similar' items/customers.
    """
    numeric_df = select_important_features(df, is_unsupervised=True)
    
    # Normalize features so similarity isn't biased by large numbers
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    similarity = cosine_similarity(scaled_data)

    results = {}
    # Sample top 5 records to show recommendations for
    for idx in range(min(5, len(similarity))):
        # Sort by similarity, exclude the item itself [1:]
        similar_indices = np.argsort(similarity[idx])[::-1][1:top_n + 1]
        scores = np.sort(similarity[idx])[::-1][1:top_n + 1]
        results[f"Record_{idx}"] = {
            "matches": similar_indices.tolist(),
            "confidence_scores": np.round(scores, 3).tolist()
        }
    acc = validate_model_performance(scaled_data, np.arange(len(scaled_data)), "regression")
    return {
        "model": "Normalized Cosine Similarity",
        "recommendations": results,
        "model_confidence": acc
    }

# -------------------------------
# 4. Price Optimization / Risk (Impact Analysis)
# -------------------------------
def price_optimization(df):
    """
    Refactored: Uses Regularized Regression (Ridge) to ensure 
    feature impact scores are stable and not exaggerated.
    """
    from sklearn.linear_model import Ridge
    
    numeric_df = select_important_features(df, is_unsupervised=False)
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]
    
    # Standardize X for fair coefficient comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)

    # Impact is the coefficient of the scaled feature
    impact = dict(zip(X.columns, model.coef_))
    sorted_impact = dict(sorted(impact.items(), key=lambda item: abs(item[1]), reverse=True))
    acc = validate_model_performance(X_scaled, y, "regression")
    return {
        "model": "Ridge Regression (Impact Analysis)",
        "target": y.name,
        "feature_influence_score": {k: round(v, 4) for k, v in sorted_impact.items()},
        "insight": "Higher absolute values indicate stronger influence on target.",
        "model_confidence": acc
    }

# -------------------------------
# 5. Anomaly Detection
# -------------------------------
def anomaly_detection(df):
    """
    Refactored: Optimized Isolation Forest with automated 
    contamination estimation based on dataset size.
    """
    numeric_df = select_important_features(df, is_unsupervised=True)
    
    # Auto-adjust contamination if data is very small
    contam = 0.05 if len(numeric_df) > 50 else 0.1

    model = IsolationForest(
        n_estimators=200, # Increased for better stability
        contamination=contam,
        random_state=42
    )

    anomalies = model.fit_predict(numeric_df)
    
    # Convert labels: 1 (Normal) -> "Clean", -1 (Anomaly) -> "Flagged"
    results = pd.Series(anomalies).map({1: "Normal", -1: "Anomaly"}).value_counts().to_dict()
    acc = validate_model_performance(numeric_df, anomalies, "classification")
    return {
        "model": "Isolation Forest (Optimized)",
        "distribution": results,
        "anomaly_indices": np.where(anomalies == -1)[0].tolist(),
        "model_confidence": acc
    }
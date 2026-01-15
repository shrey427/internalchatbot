import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
import streamlit as st
import streamlit_app

# Assume uploaded_file comes from existing Streamlit UI
# uploaded_file = st.file_uploader(...)  # Already handled in your code

if streamlit_app.sales_file is not None:
    # Read uploaded CSV
    df = pd.read_csv(streamlit_app.sales_file)

    # Check if 'amount' column exists
    if "amount" not in df.columns:
        st.error("❌ CSV must contain an 'amount' column")
    else:
        X = df[["amount"]]

        # Train Isolation Forest model
        model = IsolationForest(
            n_estimators=150,
            contamination=0.05,
            random_state=42
        )
        model.fit(X)

        # Save the trained model
        save_path = "finance_model.pkl"
        joblib.dump(model, save_path)
        st.success(f"✅ Finance anomaly model saved as {save_path}")
else:
    st.info("📂 No file uploaded yet")

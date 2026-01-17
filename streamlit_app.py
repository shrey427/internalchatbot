import streamlit as st
import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

# Import ML models
from ml_models.sales_models import (
    sales_forecasting, customer_segmentation, 
    recommendation_system, price_optimization, anomaly_detection as sales_anomaly
)
from ml_models.finance_models import (
    credit_risk_assessment, expense_budget_forecasting,
    portfolio_optimization, invoice_anomaly_detection, financial_statement_analysis
)
from ml_models.hr_models import (
    attrition_turnover_prediction, workforce_planning,
    training_recommendation, salary_compensation_benchmarking
)

# -------------------------------
# Environment & LLM Setup
# -------------------------------
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN not found in environment variables.")
    st.stop()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

MODEL_NAME = "deepseek-ai/DeepSeek-R1:novita"

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="BI Multi-Domain Assistant",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Helper: Smart Context Fetcher
# -------------------------------
def get_smart_context(df, ml_results, query):
    """
    Extracts relevant rows (Anomalies + Outliers + Query-Matches)
    to provide the LLM a representative 'peek' at the entire dataset.
    """
    relevant_indices = set()

    # 1. Capture ML Anomalies / Flagged Records
    if isinstance(ml_results, dict):
        if "anomaly_indices" in ml_results:
            relevant_indices.update(ml_results["anomaly_indices"])
        if "flagged_record_indices" in ml_results:
            relevant_indices.update(ml_results["flagged_record_indices"])
        if "recommendations" in ml_results:
            # For HR recommendation peers
            for k, v in ml_results["recommendations"].items():
                relevant_indices.update(v.get("peer_group_matches", []))

    # 2. Intent-Based Filtering (e.g., 'losses', 'negative', 'stress')
    query_lc = query.lower()
    if any(word in query_lc for word in ["loss", "negative", "stress", "debt", "risk", "decline"]):
        num_df = df.select_dtypes(include='number')
        neg_mask = (num_df < 0).any(axis=1)
        relevant_indices.update(df[neg_mask].index.tolist())

    # 3. Add Top/Bottom Outliers
    numeric_cols = df.select_dtypes(include='number').columns
    if not numeric_cols.empty:
        main_col = numeric_cols[-1]
        relevant_indices.update(df[main_col].nlargest(5).index.tolist())
        relevant_indices.update(df[main_col].nsmallest(5).index.tolist())

    # 4. Final Construction (Head + Relevant Rows)
    final_indices = list(relevant_indices)[:30] # Limit to 30 rows for token efficiency
    context_df = pd.concat([df.head(5), df.iloc[final_indices]]).drop_duplicates()
    
    return context_df.to_string(index=False)

# -------------------------------
# Sidebar – Domain & Use Case Controls
# -------------------------------
st.sidebar.title("🏢 Business Department")
domain = st.sidebar.selectbox("Choose Department", ["Sales", "Finance", "HR"])

use_cases = {
    "Sales": [ "Sales Forecasting", "Customer Segmentation", 
        "Recommendation System", "Price Optimization", "Anomaly Detection"],
    "Finance": ["Credit Scoring / Risk Assessment", "Expense / Budget Forecasting",
        "Portfolio Optimization", "Invoice / Payment Anomaly Detection", "Financial Statement Analysis"],
    "HR": [ "Attrition / Turnover Prediction", "Workforce Planning",
        "Training Recommendation", "Salary / Compensation Benchmarking"]
}

selected_use_case = st.sidebar.selectbox(f"Select {domain} Use Case", use_cases[domain])

# --- Integrated Reset Logic ---

# Initialize state trackers if they don't exist
if "current_domain" not in st.session_state:
    st.session_state.current_domain = domain
if "current_use_case" not in st.session_state:
    st.session_state.current_use_case = selected_use_case

# AUTO-CLEAR: Detect if Department or Use Case has changed
if (st.session_state.current_domain != domain or 
    st.session_state.current_use_case != selected_use_case):
    
    st.session_state.chat_history = []  # Wipe the chat
    st.session_state.current_domain = domain  # Update trackers
    st.session_state.current_use_case = selected_use_case
    st.rerun()  # Refresh UI to show clean slate

# MANUAL CLEAR: Button in Sidebar
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# -------------------------------
# Main UI
# -------------------------------
st.title(f"AI Powered Business Intelligence Assistant")
st.subheader(f"Get real-time data-driven insights for {domain} use cases")
# File Uploader
uploaded_file = st.sidebar.file_uploader(f"Upload {domain} Data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Handle Excel vs CSV
        if uploaded_file.name.endswith('.xlsx'):
            xl = pd.ExcelFile(uploaded_file)
            sheet_name = st.sidebar.selectbox("Select Sheet", xl.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        else:
            df = pd.read_csv(uploaded_file)

        st.success("✅ Data Loaded Successfully!")
        
        with st.expander("📊 Preview Raw Data"):
            st.dataframe(df.head(10))

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display Chat History
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(text)

        # Chat Input Logic
        if query := st.chat_input("Ask a question about your data..."):
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.chat_history.append(("user", query))

            with st.spinner("Analyzing data and generating insights..."):
                # 1. Run Backend ML Logic
                ml_results = {}
                if domain == "Sales":
                    if "Forecast" in selected_use_case: ml_results = sales_forecasting(df)
                    elif "Segmentation" in selected_use_case: ml_results = customer_segmentation(df)
                    elif "Recommendation" in selected_use_case: ml_results = recommendation_system(df)
                    elif "Price" in selected_use_case: ml_results = price_optimization(df)
                    elif "Anomaly" in selected_use_case: ml_results = sales_anomaly(df)
                elif domain == "Finance":
                    if "Credit Risk" in selected_use_case: ml_results = credit_risk_assessment(df)
                    elif "Expense Forecast" in selected_use_case: ml_results = expense_budget_forecasting(df)
                    elif "Portfolio" in selected_use_case: ml_results = portfolio_optimization(df)
                    elif "Invoice" in selected_use_case: ml_results = invoice_anomaly_detection(df)
                    elif "Statement" in selected_use_case: ml_results = financial_statement_analysis(df)
                elif domain == "HR":
                    if "Attrition" in selected_use_case: ml_results = attrition_turnover_prediction(df)
                    elif "Workforce" in selected_use_case: ml_results = workforce_planning(df)
                    elif "Training" in selected_use_case: ml_results = training_recommendation(df)
                    elif "Salary" in selected_use_case: ml_results = salary_compensation_benchmarking(df)

                # 2. Get Smart Context (The "Peek")
                smart_data_peek = get_smart_context(df, ml_results, query)

                # 3. Fine-Tuned Prompt Generation
                prompt = f"""
                ### ROLE
                You are a Senior {domain} Strategy Consultant. Provide executive-grade insights.

                ### DATA CONTEXT (SMART PEEK)
                {smart_data_peek}

                ### ML MODEL RESULTS
                {ml_results}

                ### USER QUESTION
                {query}

                ### STRICTURES
                1. NO CHAIN OF THOUGHT: Do not include <think> tags.
                2. NO FILLER: Start immediately with headers.
                3. USE NAMES: If names/IDs are in the DATA CONTEXT, use them to answer row-specific questions.
                4. INTEGRATE: Combine the raw data peek with the ML results for a unified answer.

                ### OUTPUT STRUCTURE
                ### Business Insights
                - [Analyze specific trends or outliers found in the data peek and ML results]
                
                ### Strategic Recommendations
                - [Actionable steps based on the findings]
                """

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": f"You are a Senior {domain} Consultant specializing in data-driven reporting."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
            # Use Regex to remove everything between <tool_call> and <tool_call> tags
            raw_answer = response.choices[0].message.content
            
            def extract_business_output(text: str) -> str:
                markers = ["Business Insights", "Strategic Recommendations"]

                for marker in markers:
                    if marker in text:
                        return text[text.index(marker):].strip()

                return text  # fallback 
            
            answer =  extract_business_output(raw_answer)
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat_history.append(("assistant",answer))
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info(f"👈 Please upload a file to begin.")
import streamlit as st
import pandas as pd
import numpy as np
from pandas.errors import EmptyDataError
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

# --- IMPORT MODELS AND DECORATOR ---
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
from ml_models.utils import with_backend_peek # Import your new utility

# --- WRAP MODELS WITH BACKEND PEEKING (DOES NOT MODIFY ORIGINAL FILES) ---
# This ensures every model now returns 'backend_context' automatically
sales_forecasting = with_backend_peek(sales_forecasting)
customer_segmentation = with_backend_peek(customer_segmentation)
recommendation_system = with_backend_peek(recommendation_system)
price_optimization = with_backend_peek(price_optimization)
sales_anomaly = with_backend_peek(sales_anomaly)

credit_risk_assessment = with_backend_peek(credit_risk_assessment)
expense_budget_forecasting = with_backend_peek(expense_budget_forecasting)
portfolio_optimization = with_backend_peek(portfolio_optimization)
invoice_anomaly_detection = with_backend_peek(invoice_anomaly_detection)
financial_statement_analysis = with_backend_peek(financial_statement_analysis)

attrition_turnover_prediction = with_backend_peek(attrition_turnover_prediction)
workforce_planning = with_backend_peek(workforce_planning)
training_recommendation = with_backend_peek(training_recommendation)
salary_compensation_benchmarking = with_backend_peek(salary_compensation_benchmarking)

# -------------------------------
# Environment & LLM Setup
# -------------------------------
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN not found in environment variables.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=HF_TOKEN,
)

MODEL_NAME = "tngtech/deepseek-r1t2-chimera:free"

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="BI Multi-Domain Assistant",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# Helper Functions
# -------------------------------
def sanitize_llm_output(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    pattern = r"(###\s*Business Insights[\s\S]*?###\s*Strategic Recommendations[\s\S]*)"
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    return matches[-1].strip() if matches else text

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

# State Trackers & Reset Logic
if "current_domain" not in st.session_state: st.session_state.current_domain = domain
if "current_use_case" not in st.session_state: st.session_state.current_use_case = selected_use_case

if (st.session_state.current_domain != domain or st.session_state.current_use_case != selected_use_case):
    st.session_state.chat_history = []
    st.session_state.current_domain = domain
    st.session_state.current_use_case = selected_use_case
    st.rerun()

if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# -------------------------------
# Main UI
# -------------------------------
st.title(f"AI Powered Business Intelligence Assistant")
uploaded_file = st.sidebar.file_uploader(f"Upload {domain} Data", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.xlsx'):
            xl = pd.ExcelFile(uploaded_file)
            sheet_name = st.sidebar.selectbox("Select Sheet", xl.sheet_names)
            df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
        else:
            df = pd.read_csv(uploaded_file)

        st.success("✅ Data Loaded Successfully!")
        
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for role, text in st.session_state.chat_history:
            with st.chat_message(role): st.markdown(text)

        if query := st.chat_input("Ask a question about your data..."):
            with st.chat_message("user"): st.markdown(query)
            st.session_state.chat_history.append(("user", query))

            with st.spinner("Analyzing data..."):
                # 1. Run Backend ML Logic (Now includes automated peeking)
                ml_results = {}
                if domain == "Sales":
                    if "Forecast" in selected_use_case: ml_results = sales_forecasting(df)
                    elif "Segmentation" in selected_use_case: ml_results = customer_segmentation(df)
                    elif "Recommendation" in selected_use_case: ml_results = recommendation_system(df)
                    elif "Price" in selected_use_case: ml_results = price_optimization(df)
                    elif "Anomaly" in selected_use_case: ml_results = sales_anomaly(df)
                elif domain == "Finance":
                    if "Credit Scoring" in selected_use_case: ml_results = credit_risk_assessment(df)
                    elif "Expense Forecast" in selected_use_case: ml_results = expense_budget_forecasting(df)
                    elif "Portfolio" in selected_use_case: ml_results = portfolio_optimization(df)
                    elif "Invoice" in selected_use_case: ml_results = invoice_anomaly_detection(df)
                    elif "Statement" in selected_use_case: ml_results = financial_statement_analysis(df)
                elif domain == "HR":
                    if "Attrition" in selected_use_case: ml_results = attrition_turnover_prediction(df)
                    elif "Workforce" in selected_use_case: ml_results = workforce_planning(df)
                    elif "Training" in selected_use_case: ml_results = training_recommendation(df)
                    elif "Salary" in selected_use_case: ml_results = salary_compensation_benchmarking(df)

                # 2. UPDATED PROMPT: Uses Backend Context for superior accuracy
                # We separate context (from backend) from results (from model)
                backend_ctx = ml_results.get("backend_context", {})
                
                prompt = f"""
                ### ROLE
                You are a Senior {domain} Strategy Consultant. 

                ### BACKEND DATA CONTEXT
                - AVAILABLE COLUMNS: {backend_ctx.get('columns_present', 'Not provided')}
                - EVIDENCE ROWS (Representative sample): {backend_ctx.get('evidence_rows', [])}
                - STATISTICAL OVERVIEW: {backend_ctx.get('data_summary', {})}

                ### ML ANALYTICS RESULTS
                { {k:v for k,v in ml_results.items() if k != 'backend_context'} }

                ### USER QUESTION
                {query}

                ### STRICTURES
                1. NO THINKING: Do not include <think> tags.
                2. USE EVIDENCE: Reference specific names, IDs, or values from the 'EVIDENCE ROWS' to support your points.
                3. SCHEMA AWARENESS: Mention specific columns if they are relevant to the user's question.
                4. INTEGRATE: Use the statistical overview to provide context for the ML results.

                ### OUTPUT STRUCTURE
                ### Business Insights
                - [Analyze specific findings using evidence rows and ML results]
                
                ### Strategic Recommendations
                - [Provide 2-3 actionable next steps]
                """

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": f"You are a Senior {domain} Consultant specializing in data-driven reporting."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
            raw_answer = response.choices[0].message.content
            answer = sanitize_llm_output(raw_answer)
            with st.chat_message("assistant"): st.markdown(answer)
            st.session_state.chat_history.append(("assistant",answer))

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info(f"👈 Please upload a file to begin.")
import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
from openai import OpenAI
from dotenv import load_dotenv
import os

# Import ML models from your local folder structure
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
    st.error("❌ HF_TOKEN not found in environment variables.")
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
# Sidebar – Domain & Use Case Controls
# -------------------------------
st.sidebar.title("🏢 Business Department")
domain = st.sidebar.selectbox("Select Domain", ["Sales", "Finance", "HR"])

st.sidebar.markdown("---")
st.sidebar.subheader(f"📁 {domain} Data Upload")
uploaded_file = st.sidebar.file_uploader(f"Upload {domain} CSV", type="csv")

# Dynamic Use Case Mapping based on provided images
use_case_map = {
    "Sales": [
        "Sales Forecasting", "Customer Segmentation", 
        "Recommendation System", "Price Optimization", "Anomaly Detection"
    ],
    "Finance": [
        "Credit Scoring / Risk Assessment", "Expense / Budget Forecasting",
        "Portfolio Optimization", "Invoice / Payment Anomaly Detection", "Financial Statement Analysis"
    ],
    "HR": [
        "Attrition / Turnover Prediction", "Workforce Planning",
        "Training Recommendation", "Salary / Compensation Benchmarking"
    ]
}

selected_use_case = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Select a use case"] + use_case_map[domain]
)

# -------------------------------
# Model Routing Logic
# -------------------------------
def run_ml_logic(domain_name, use_case, data):
    """
    Routes data to the correct backend function based on domain and use case.
    """
    if domain_name == "Sales":
        if use_case == "Sales Forecasting": return sales_forecasting(data)
        if use_case == "Customer Segmentation": return customer_segmentation(data)
        if use_case == "Recommendation System": return recommendation_system(data)
        if use_case == "Price Optimization": return price_optimization(data)
        if use_case == "Anomaly Detection": return sales_anomaly(data)

    elif domain_name == "Finance":
        if use_case == "Credit Scoring / Risk Assessment": return credit_risk_assessment(data)
        if use_case == "Expense / Budget Forecasting": return expense_budget_forecasting(data)
        if use_case == "Portfolio Optimization": return portfolio_optimization(data)
        if use_case == "Invoice / Payment Anomaly Detection": return invoice_anomaly_detection(data)
        if use_case == "Financial Statement Analysis": return financial_statement_analysis(data)

    elif domain_name == "HR":
        if use_case == "Attrition / Turnover Prediction": return attrition_turnover_prediction(data)
        if use_case == "Workforce Planning": return workforce_planning(data)
        if use_case == "Training Recommendation": return training_recommendation(data)
        if use_case == "Salary / Compensation Benchmarking": return salary_compensation_benchmarking(data)

    return {}

# -------------------------------
# Main Interface
# -------------------------------
st.title(f"AI Powered Business Intelligence Assistant")
st.subheader(f"Get real-time data-driven insights for {domain} use cases")

if uploaded_file and selected_use_case != "Select a use case":
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Ready: {selected_use_case}")
        
        # Chat Interface
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Display Chat History
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        # User Input
        if query := st.chat_input("Ask for insights..."):
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.chat_history.append(("user", query))

            with st.spinner("🤖generating insights..."):
                # 1. Trigger ML Backend
                ml_results = run_ml_logic(domain, selected_use_case, df)
                
                # 2. LLM Insight Generation
                prompt = f"""
                Domain: {domain}
                Use Case: {selected_use_case}
                ML Outputs: {ml_results}
                User Question: {query}
                
                You are a Senior Business Intelligence Strategy Consultant. Your goal is to translate complex ML outputs into high-level executive actions.Follow below structure strictly for output generation

                ### STRUCTURES (CRITICAL)
                1. NO CHAIN OF THOUGHT: Do not include  tags or any internal reasoning steps.
                3. NO PREAMBLE/POSTAMBLE: Start immediately with the first header and end immediately after the last recommendation.
                4. DATA INTEGRITY: Use only the numbers provided in the ML results. Do not hallucinate external market trends.

                ### OUTPUT STRUCTURE
                ### Business Insights
                - [Insight 1: Describe a specific data relationship or trend found in the ML results]
                - [Insight 2: Identify a potential risk or opportunity indicated by the metrics]

                ### Strategic Recommendations
                - [Action 1: Immediate operational step based on Insight 1]
                - [Action 2: Long-term strategic adjustment based on Insight 2]

                """
                
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": f"You are a Senior {domain} Consultant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                answer = response.choices[0].message.content

            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.chat_history.append(("assistant", answer))

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info(f"👈 Please upload a {domain} CSV file and select a use case to begin.")
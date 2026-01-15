import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
from dotenv import load_dotenv
import os
import plotly.express as px

# -------------------------------
# Environment & LLM Setup
# -------------------------------
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found. Please set it as an environment variable.")

client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=HF_TOKEN)
MODEL_NAME = "deepseek-ai/DeepSeek-R1:novita"

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI-Powered BI Assistant",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# CSS for Professional UI
# -------------------------------
st.markdown("""
<style>
/* Chat boxes */
.chat-box {
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    max-width: 80%;
}
.user { background-color: #0d6efd; color: white; margin-left: auto; text-align: right; }
.bot { background-color: #198754; color: white; margin-right: auto; text-align: left; }

/* Cards */
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #1f2937;
    color: #f1f5f9;
    margin-bottom: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

/* Sidebar headers */
.sidebar .stMarkdown h2 {
    color: #0d6efd;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar: Upload + Use Case
# -------------------------------
st.sidebar.header("📁 Upload & Configure Analysis")

domain = st.sidebar.selectbox("Select Domain", ["Sales", "Finance", "HR"])
use_cases = {
    "Sales": ["Sales Forecast", "Churn Prediction", "Customer Segmentation"],
    "Finance": ["Fraud Detection", "Expense Forecast"],
    "HR": ["Attrition Prediction", "Employee Segmentation"]
}
use_case = st.sidebar.selectbox("Select Use Case", use_cases[domain])
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

df = pd.read_csv(uploaded_file) if uploaded_file else None
if df is not None:
    st.sidebar.success(f"✅ Dataset uploaded for {use_case}")
    st.sidebar.write(df.head())

# -------------------------------
# Page Header
# -------------------------------
st.title("📊 AI-Powered Business Intelligence Assistant")
st.markdown("Ask questions about your uploaded data and receive insights & recommendations from our AI assistant.")

# -------------------------------
# ML Model Execution
# -------------------------------
def run_ml_model(df, domain, use_case):
    df_clean = df.dropna().copy()
    for col in df_clean.select_dtypes(include="object"):
        df_clean[col] = LabelEncoder().fit_transform(df_clean[col])
    
    ml_result = {}
    chart = None

    if use_case == "Churn Prediction" and "Churn" in df_clean.columns:
        X = df_clean.drop("Churn", axis=1)
        y = df_clean["Churn"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        df_clean["Prediction"] = preds
        ml_result = df_clean["Prediction"].value_counts().to_dict()
        chart = px.pie(df_clean, names="Prediction", title="Churn Prediction Distribution")

    elif use_case == "Customer Segmentation" or use_case == "Employee Segmentation":
        X = df_clean.select_dtypes(include="number")
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_clean["Segment"] = kmeans.fit_predict(X)
        ml_result = df_clean["Segment"].value_counts().to_dict()
        chart = px.bar(x=list(ml_result.keys()), y=list(ml_result.values()), labels={"x":"Segment", "y":"Count"}, title="Segments Distribution")

    elif use_case == "Fraud Detection":
        X = df_clean.select_dtypes(include="number")
        iso = IsolationForest(contamination=0.05, random_state=42)
        df_clean["Anomaly"] = iso.fit_predict(X)
        ml_result = df_clean["Anomaly"].value_counts().to_dict()
        chart = px.pie(df_clean, names="Anomaly", title="Fraud / Anomaly Distribution")

    elif use_case == "Expense Forecast" or use_case == "Sales Forecast":
        numeric_cols = df_clean.select_dtypes(include="number").columns
        ml_result = {col: float(df_clean[col].sum()) for col in numeric_cols}
    
    elif use_case == "Attrition Prediction" and "Attrition" in df_clean.columns:
        X = df_clean.drop("Attrition", axis=1)
        y = df_clean["Attrition"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        df_clean["Attrition_Prediction"] = model.predict(X)
        ml_result = df_clean["Attrition_Prediction"].value_counts().to_dict()
        chart = px.pie(df_clean, names="Attrition_Prediction", title="Attrition Prediction Distribution")
    
    return ml_result, chart

# -------------------------------
# LLM Insight Generation
# -------------------------------
def generate_llm_insights(user_query, ml_outputs, df_summary):
    prompt = f"""
You are a senior business intelligence consultant.

User Question:
{user_query}

ML Outputs:
{ml_outputs}

Data Summary:
{df_summary}

Task:
- Interpret the ML outputs
- Provide business insights
- Give strategic recommendations
- Be concise and professional
"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You provide executive-level BI insights."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# -------------------------------
# Chat Interface
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask a business question...")

if query:
    df_summary = df.head().to_dict() if df is not None else "No dataset uploaded."
    ml_outputs, chart = run_ml_model(df, domain, use_case) if df is not None else ({}, None)
    answer = generate_llm_insights(query, ml_outputs, df_summary)

    st.session_state.chat_history.append(("user", query))
    st.session_state.chat_history.append(("bot", answer))

    # Show ML chart if exists
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)

# -------------------------------
# Display Chat
# -------------------------------
for role, message in st.session_state.chat_history:
    if role == "user":
        st.markdown(f"<div class='chat-box user'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-box bot'>{message}</div>", unsafe_allow_html=True)

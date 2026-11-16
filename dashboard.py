"""
dashboard.py
============
Interactive pure-Python web dashboard built with Streamlit and Plotly.
No HTML, CSS, or JavaScript required.

Features:
1. Tab 1: Dataset Insights & Churn Statistics (KPIs, interactive Plotly charts)
2. Tab 2: Model Performance & Comparison (Metrics table, ROC curves, feature importance)
3. Tab 3: Live Customer Churn Prediction (Interactive form + real-time probability gauge)
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------
# PAGE CONFIGURATION (Pure Python Streamlit setup)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    page_icon="📡",
    layout="wide"
)

# -------------------------------------------------------------
# HELPER FUNCTIONS TO LOAD DATA AND MODELS
# -------------------------------------------------------------
@st.cache_data
def load_raw_and_summary_data():
    """Loads raw customer data and evaluation reports."""
    raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if not os.path.exists(raw_path):
        # Fallback to any CSV in data/raw
        csvs = [os.path.join("data/raw", f) for f in os.listdir("data/raw") if f.endswith(".csv")]
        raw_path = csvs[0] if csvs else None

    raw_df = pd.read_csv(raw_path) if raw_path else None
    metrics_path = "reports/test_metrics_summary.csv"
    metrics_df = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None

    return raw_df, metrics_df


@st.cache_resource
def load_model_artifacts():
    """Loads preprocessor, best model, and feature names."""
    try:
        model = joblib.load("models/best_model.joblib")
        preprocessor = joblib.load("models/preprocessor.joblib")
        model_name = joblib.load("models/best_model_name.joblib") if os.path.exists("models/best_model_name.joblib") else type(model).__name__
        feature_names = joblib.load("models/feature_names.joblib") if os.path.exists("models/feature_names.joblib") else None
        return model, preprocessor, model_name, feature_names
    except Exception:
        return None, None, "None", None


# Load cached resources
raw_df, metrics_df = load_raw_and_summary_data()
model, preprocessor, model_name, feature_names = load_model_artifacts()

# -------------------------------------------------------------
# APP HEADER
# -------------------------------------------------------------
st.title("📡 Telco Customer Churn Analytics & AI Deployment")
st.caption("A beginner-friendly machine learning pipeline that predicts customer churn, explains key risk factors, and provides actionable retention recommendations.")

# -------------------------------------------------------------
# DASHBOARD TABS
# -------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 Churn Statistics & Analytics",
    "🤖 Model Comparison & AI Insights",
    "🔮 Live Customer Churn Predictor"
])

# =============================================================
# TAB 1: CHURN STATISTICS & ANALYTICS
# =============================================================
with tab1:
    st.subheader("Dataset Overview & Key Statistics")

    if raw_df is not None:
        # KPI Metric Cards
        total_customers = len(raw_df)
        churn_count = (raw_df['Churn'] == 'Yes').sum()
        churn_rate = (churn_count / total_customers) * 100
        avg_monthly = pd.to_numeric(raw_df['MonthlyCharges'], errors='coerce').mean()
        avg_tenure = pd.to_numeric(raw_df['tenure'], errors='coerce').mean()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{total_customers:,}")
        col2.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
        col3.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")
        col4.metric("Avg Tenure", f"{avg_tenure:.1f} months")

        st.divider()

        # Two interactive Plotly charts side-by-side
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            # Donut chart for Churn Distribution
            churn_dist = raw_df['Churn'].value_counts().reset_index()
            churn_dist.columns = ['Status', 'Count']
            fig_donut = px.pie(
                churn_dist,
                names='Status',
                values='Count',
                title="<b>Customer Retention vs. Churn Breakdown</b>",
                hole=0.45,
                color='Status',
                color_discrete_map={'No': '#2E86AB', 'Yes': '#E63946'}
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        with chart_col2:
            # Bar chart for Churn by Contract Type
            contract_churn = raw_df.groupby(['Contract', 'Churn']).size().reset_index(name='Customers')
            fig_contract = px.bar(
                contract_churn,
                x='Contract',
                y='Customers',
                color='Churn',
                barmode='group',
                title="<b>Churn Rate by Contract Type</b>",
                color_discrete_map={'No': '#2E86AB', 'Yes': '#E63946'}
            )
            st.plotly_chart(fig_contract, use_container_width=True)

        # Bottom row charts
        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            # Box plot of Tenure vs Churn
            fig_tenure = px.box(
                raw_df,
                x='Churn',
                y='tenure',
                color='Churn',
                title="<b>Tenure (Months) Distribution by Churn Status</b>",
                color_discrete_map={'No': '#2E86AB', 'Yes': '#E63946'}
            )
            st.plotly_chart(fig_tenure, use_container_width=True)

        with chart_col4:
            # Bar chart of Internet Service vs Churn
            internet_churn = raw_df.groupby(['InternetService', 'Churn']).size().reset_index(name='Customers')
            fig_internet = px.bar(
                internet_churn,
                x='InternetService',
                y='Customers',
                color='Churn',
                barmode='group',
                title="<b>Churn by Internet Service Type</b>",
                color_discrete_map={'No': '#2E86AB', 'Yes': '#E63946'}
            )
            st.plotly_chart(fig_internet, use_container_width=True)

    else:
        st.warning("Raw dataset not found in 'data/raw/'. Please run preprocess.py first.")

# =============================================================
# TAB 2: MODEL COMPARISON & AI INSIGHTS
# =============================================================
with tab2:
    st.subheader("Model Evaluation & Deep Learning Comparison")

    if metrics_df is not None:
        st.markdown(f"**Current Champion Model:** `{model_name}` (Selected based on balanced F1-Score & ROC-AUC)")

        # Display Metrics Comparison Table
        st.dataframe(metrics_df, use_container_width=True)

        # Plotly Grouped Bar Chart comparing all metrics
        metrics_melted = metrics_df.melt(
            id_vars=["Model"],
            value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            var_name="Metric",
            value_name="Score"
        )
        fig_metrics = px.bar(
            metrics_melted,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            title="<b>Performance Metrics Comparison (Holdout Test Set)</b>",
            text_auto=".2f"
        )
        fig_metrics.update_layout(yaxis_range=[0, 1.0])
        st.plotly_chart(fig_metrics, use_container_width=True)

        st.divider()

        # Display Confusion Matrices & ROC Curves
        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.markdown("#### Confusion Matrices (2x2 Grid)")
            if os.path.exists("reports/confusion_matrices.png"):
                st.image("reports/confusion_matrices.png", use_container_width=True)
            else:
                st.info("Run 'python evaluate.py' to generate confusion matrix plots.")

        with img_col2:
            st.markdown("#### ROC Curves Comparison")
            if os.path.exists("reports/roc_curves.png"):
                st.image("reports/roc_curves.png", use_container_width=True)
            else:
                st.info("Run 'python evaluate.py' to generate ROC curve plots.")

        st.divider()

        # Feature Importance & Deep Learning Tuning
        feat_col1, feat_col2 = st.columns(2)

        with feat_col1:
            st.markdown("#### Top Churn Drivers (Random Forest)")
            if os.path.exists("reports/feature_importance.png"):
                st.image("reports/feature_importance.png", use_container_width=True)
            st.markdown("""
            **Top 5 Churn Drivers Explained:**
            1. **TotalCharges & MonthlyCharges:** Higher charges increase price sensitivity and churn likelihood.
            2. **Tenure:** Newer customers (months 1-6) have weak brand loyalty and churn most frequently.
            3. **MonthlyToTotalRatio:** Sharp spikes in recent bills relative to history trigger churn.
            4. **Contract Type:** Month-to-month contracts have 4x higher churn than 1 or 2-year contracts.
            5. **Security & Tech Support:** Add-on services increase friction, keeping customers loyal.
            """)

        with feat_col2:
            st.markdown("#### Deep Learning (MLPClassifier) Tuning")
            if os.path.exists("reports/mlp_tuning_results.csv"):
                mlp_df = pd.read_csv("reports/mlp_tuning_results.csv")
                st.dataframe(mlp_df, use_container_width=True)
                st.markdown("""
                **Key Takeaway from Deep Learning on Tabular Data:**
                - The neural network (MLPClassifier) achieved strong accuracy (~80.1%), but **Random Forest** achieved superior recall (77.3% vs 50.0%) and higher F1-score for catching churners.
                - For tabular business data with high class imbalance, tree ensembles with balanced class weights generally outperform shallow neural networks.
                """)
    else:
        st.warning("Evaluation metrics not found. Please run 'python evaluate.py' to generate reports.")

# =============================================================
# TAB 3: LIVE CUSTOMER CHURN PREDICTION
# =============================================================
with tab3:
    st.subheader("Live Customer Churn Risk Prediction")
    st.markdown("Adjust the customer's contract, tenure, and subscribed services below to get an instant prediction.")

    if model is None or preprocessor is None:
        st.error("Model artifacts not found. Please run 'python preprocess.py', 'python train.py', and 'python evaluate.py' first.")
    else:
        # Input Form
        with st.form("churn_prediction_form"):
            form_col1, form_col2, form_col3 = st.columns(3)

            with form_col1:
                st.markdown("**Personal Details**")
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["No", "Yes"])
                tenure = st.slider("Tenure (Months with company)", min_value=0, max_value=72, value=4)

            with form_col2:
                st.markdown("**Services Subscribed**")
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
                security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

            with form_col3:
                st.markdown("**Contract & Billing**")
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                device_protect = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                monthly = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=85.0, step=1.0)
                total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly), step=10.0)

            submitted = st.form_submit_button("⚡ Predict Churn Probability", type="primary", use_container_width=True)

        if submitted:
            # 1. Package inputs into DataFrame
            sample_data = pd.DataFrame([{
                'gender': gender,
                'SeniorCitizen': senior,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone,
                'MultipleLines': multiple,
                'InternetService': internet,
                'OnlineSecurity': security,
                'OnlineBackup': backup,
                'DeviceProtection': device_protect,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless,
                'PaymentMethod': payment,
                'MonthlyCharges': monthly,
                'TotalCharges': total
            }])

            # 2. Engineer features
            def get_tenure_group(t):
                if t <= 12:
                    return "0-12m"
                elif t <= 24:
                    return "13-24m"
                elif t <= 48:
                    return "25-48m"
                else:
                    return "49-72m"

            sample_data['TenureGroup'] = sample_data['tenure'].apply(get_tenure_group)

            service_cols = [
                'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
            ]
            sample_data['TotalServices'] = sample_data[service_cols].apply(
                lambda row: sum(row[col] == 'Yes' for col in service_cols), axis=1
            )
            sample_data['MonthlyToTotalRatio'] = sample_data['MonthlyCharges'] / (sample_data['TotalCharges'] + 1.0)

            # 3. Transform using preprocessor
            transformed_input = preprocessor.transform(sample_data)
            if feature_names is not None:
                transformed_input = pd.DataFrame(transformed_input, columns=feature_names)

            # 4. Predict
            pred = int(model.predict(transformed_input)[0])
            proba = float(model.predict_proba(transformed_input)[0][1])
            proba_percent = proba * 100

            st.divider()

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("### Prediction Result")
                if pred == 1:
                    st.error(f"🚨 **High Risk of Churn!** (Probability: {proba_percent:.1f}%)")
                    st.markdown("""
                    **Recommended Immediate Actions:**
                    - 🏷️ Offer a proactive 15% discount on an annual contract.
                    - 🛠️ Bundle free Tech Support or Online Security to increase stickiness.
                    - 📞 Trigger a retention outreach call from customer support.
                    """)
                else:
                    st.success(f"✅ **Customer is Likely to Stay** (Probability of Churn: {proba_percent:.1f}%)")
                    st.markdown("""
                    **Recommended Strategy:**
                    - 🎁 Reward loyalty with occasional perks or loyalty points.
                    - 📺 Explore cross-selling premium entertainment bundles.
                    """)

            with res_col2:
                # Plotly Gauge Chart for Churn Probability
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=proba_percent,
                    title={'text': "<b>Churn Probability Gauge</b>", 'font': {'size': 18}},
                    number={'suffix': "%"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#D90429" if proba_percent >= 50 else "#2B9348"},
                        'steps': [
                            {'range': [0, 35], 'color': "#D8F3DC"},
                            {'range': [35, 60], 'color': "#FFF3B0"},
                            {'range': [60, 100], 'color': "#FFCCD5"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

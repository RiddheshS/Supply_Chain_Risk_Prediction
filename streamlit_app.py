import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# --- Robust auto-create model on first run (works on Streamlit Cloud) ----------
MODEL_PATH = Path("ml_model/risk_prediction_model.pkl")
if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    # Try normal package import first
    try:
        import ml_model.train_model as trainer
    except ModuleNotFoundError:
        # Fallback: import by file path so it works even without __init__.py
        import importlib.util
        train_path = Path("ml_model") / "train_model.py"
        spec = importlib.util.spec_from_file_location("trainer", train_path)
        trainer = importlib.util.module_from_spec(spec)
        assert spec and spec.loader, "Cannot load trainer spec"
        spec.loader.exec_module(trainer)

    trainer.train_model()  # must save to ml_model/risk_prediction_model.pkl
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Supply Chain Risk Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    model_path = 'ml_model/risk_prediction_model.pkl'
    return joblib.load(model_path)

def init_db():
    conn = sqlite3.connect('supply_chain_risk.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS risk_predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  supplier_id TEXT,
                  product_id TEXT,
                  quantity REAL,
                  price REAL,
                  delivery_time REAL,
                  supplier_reliability REAL,
                  demand_volatility REAL,
                  geopolitical_risk REAL,
                  economic_indicators REAL,
                  historical_delays REAL,
                  risk_level INTEGER,
                  risk_score REAL,
                  timestamp TEXT)''')
    conn.commit()
    return conn

def save_prediction(data, risk_level, risk_score):
    conn = init_db()
    c = conn.cursor()
    c.execute('''INSERT INTO risk_predictions
                 (supplier_id, product_id, quantity, price, delivery_time,
                  supplier_reliability, demand_volatility, geopolitical_risk,
                  economic_indicators, historical_delays, risk_level, risk_score, timestamp)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (data['supplier_id'], data['product_id'], data['quantity'], data['price'],
               data['delivery_time'], data['supplier_reliability'], data['demand_volatility'],
               data['geopolitical_risk'], data['economic_indicators'], data['historical_delays'],
               risk_level, risk_score, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_history():
    conn = init_db()
    df = pd.read_sql_query("SELECT * FROM risk_predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def predict_risk(model, input_data):
    features = np.array([[
        input_data['supplier_reliability'],
        input_data['demand_volatility'],
        input_data['geopolitical_risk'],
        input_data['economic_indicators'],
        input_data['historical_delays']
    ]])
    risk_level = int(model.predict(features)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        class_to_idx = {c: i for i, c in enumerate(getattr(model, "classes_", [0,1,2]))}
        risk_score = float(proba[class_to_idx.get(risk_level, 0)])
    else:
        risk_score = 0.0
    return risk_level, risk_score

def main():
    st.title("📊 Supply Chain Risk Prediction Dashboard")
    st.markdown("Predict and monitor supply chain risks using machine learning")
    model = load_model()

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Choose a page:", ["Risk Prediction", "Risk History", "Analytics"])

    if page == "Risk Prediction":
        show_prediction_page(model)
    elif page == "Risk History":
        show_history_page()
    elif page == "Analytics":
        show_analytics_page()

def show_prediction_page(model):
    st.header("🔮 Risk Prediction")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Parameters")
        with st.form("prediction_form"):
            supplier_id = st.text_input("Supplier ID", value="SUP001")
            product_id = st.text_input("Product ID", value="PROD001")

            col_a, col_b = st.columns(2)
            with col_a:
                quantity = st.number_input("Quantity", min_value=1, value=100)
                price = st.number_input("Price ($)", min_value=0.01, value=50.0, step=0.01)
            with col_b:
                delivery_time = st.number_input("Delivery Time (days)", min_value=1, value=7)

            st.markdown("---")
            st.subheader("Risk Factors (0-1 scale)")
            supplier_reliability = st.slider("Supplier Reliability", 0.0, 1.0, 0.8, 0.01)
            demand_volatility    = st.slider("Demand Volatility", 0.0, 1.0, 0.3, 0.01)
            geopolitical_risk    = st.slider("Geopolitical Risk", 0.0, 1.0, 0.2, 0.01)
            economic_indicators  = st.slider("Economic Indicators", 0.0, 1.0, 0.4, 0.01)
            historical_delays    = st.slider("Historical Delays", 0.0, 1.0, 0.1, 0.01)

            submitted = st.form_submit_button("Predict Risk")

            if submitted:
                input_data = {
                    'supplier_id': supplier_id,
                    'product_id': product_id,
                    'quantity': quantity,
                    'price': price,
                    'delivery_time': delivery_time,
                    'supplier_reliability': supplier_reliability,
                    'demand_volatility': demand_volatility,
                    'geopolitical_risk': geopolitical_risk,
                    'economic_indicators': economic_indicators,
                    'historical_delays': historical_delays
                }

                with st.spinner("Analyzing risk factors..."):
                    risk_level, risk_score = predict_risk(model, input_data)

                save_prediction(input_data, risk_level, risk_score)
                st.success("Risk prediction completed!")

    with col2:
        if 'risk_level' in locals():
            st.subheader("Prediction Results")
            risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            risk_colors = {0: "green", 1: "orange", 2: "red"}
            risk_color = risk_colors[risk_level]
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: {risk_color}; color: white; text-align: center;">
                <h2>{risk_labels[risk_level]}</h2>
                <h3>Risk Score: {risk_score:.4f}</h3>
            </div>
            """, unsafe_allow_html=True)

            st.subheader("Risk Factor Analysis")
            factors = {
                "Supplier Reliability": supplier_reliability,
                "Demand Volatility": demand_volatility,
                "Geopolitical Risk": geopolitical_risk,
                "Economic Indicators": economic_indicators,
                "Historical Delays": historical_delays
            }
            factor_df = pd.DataFrame(list(factors.items()), columns=['Factor', 'Score'])
            fig = px.bar(factor_df, x='Factor', y='Score',
                         title="Risk Factor Scores",
                         color='Score',
                         color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Recommendations")
            if risk_level == 0:
                st.success("✅ Low risk - Proceed with normal operations")
            elif risk_level == 1:
                st.warning("⚠️ Medium risk - Consider mitigation strategies")
                st.info("- Monitor supplier performance closely")
                st.info("- Consider backup suppliers")
                st.info("- Review delivery schedules")
            else:
                st.error("🚨 High risk - Immediate action required")
                st.error("- Seek alternative suppliers")
                st.error("- Implement contingency plans")
                st.error("- Review all contracts")

def show_history_page():
    st.header("📋 Risk History")
    df = get_history()
    if df.empty:
        st.info("No risk predictions found. Make some predictions first!")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Level",
            options=[0, 1, 2],
            default=[0, 1, 2],
            format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x]
        )
    with col2:
        supplier_filter = st.text_input("Filter by Supplier ID")
    with col3:
        date_filter = st.date_input("Filter from date", value=None)

    filtered_df = df.copy()
    if risk_filter:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]
    if supplier_filter:
        filtered_df = filtered_df[filtered_df['supplier_id'].str.contains(supplier_filter, case=False)]
    if date_filter:
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
        filtered_df = filtered_df['timestamp'].dt.date >= date_filter

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions", len(filtered_df))
    with col2:
        avg_risk = filtered_df['risk_score'].mean()
        st.metric("Average Risk Score", f"{avg_risk:.3f}")
    with col3:
        high_risk_count = len(filtered_df[filtered_df['risk_level'] == 2])
        st.metric("High Risk Cases", high_risk_count)
    with col4:
        latest_score = filtered_df['risk_score'].iloc[0] if not filtered_df.empty else 0
        st.metric("Latest Risk Score", f"{latest_score:.3f}")

    st.subheader("Prediction History")
    display_df = df.copy()
    display_df['risk_level'] = display_df['risk_level'].map({0: "Low", 1: "Medium", 2: "High"})
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    st.dataframe(
        display_df[['timestamp', 'supplier_id', 'product_id', 'quantity',
                    'price', 'risk_level', 'risk_score']].head(50),
        use_container_width=True
    )

    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name="risk_predictions.csv",
            mime="text/csv"
        )

def show_analytics_page():
    st.header("📈 Risk Analytics")
    df = get_history()
    if df.empty:
        st.info("No data available for analytics. Make some predictions first!")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Risk Level Distribution")
        risk_counts = df['risk_level'].value_counts().sort_index()
        risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        fig = px.pie(values=risk_counts.values,
                     names=[risk_labels[i] for i in risk_counts.index],
                     title="Risk Level Distribution",
                     color_discrete_sequence=['green', 'orange', 'red'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Risk Score Trend")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp')
        fig = px.line(df_sorted, x='timestamp', y='risk_score',
                      title="Risk Score Over Time", markers=True)
        fig.update_layout(xaxis_title="Date", yaxis_title="Risk Score")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Risk Factors Correlation")
    numeric_cols = ['supplier_reliability', 'demand_volatility', 'geopolitical_risk',
                    'economic_indicators', 'historical_delays', 'risk_score']
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="Correlation Matrix of Risk Factors")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Risky Suppliers")
    supplier_risks = df.groupby('supplier_id')['risk_score'].agg(['mean', 'count']).round(3)
    supplier_risks = supplier_risks.sort_values('mean', ascending=False).head(10)
    fig = px.bar(supplier_risks, x=supplier_risks.index, y='mean',
                 title="Average Risk Score by Supplier",
                 labels={'mean': 'Average Risk Score', 'supplier_id': 'Supplier ID'})
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sqlite3
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Supply Chain Risk Management",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.streamlit.io',
        'Report a bug': "https://www.streamlit.io",
        'About': "# Supply Chain Risk Prediction Dashboard\nPredict and monitor supply chain risks using machine learning."
    }
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
    }
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    }
    .form-container {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    .sidebar-content {
        padding: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .recommendation-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .rec-low { border-left-color: #10b981; }
    .rec-medium { border-left-color: #f59e0b; }
    .rec-high { border-left-color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'ml_model/risk_prediction_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("ML model not found! Please run train_model.py first.")
        return None

# Database functions
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

# Risk prediction function
def predict_risk(model, input_data):
    # Prepare features for model
    features = np.array([[
        input_data['supplier_reliability'],
        input_data['demand_volatility'],
        input_data['geopolitical_risk'],
        input_data['economic_indicators'],
        input_data['historical_delays']
    ]])

    # Get prediction
    risk_level = model.predict(features)[0]
    risk_score = model.predict_proba(features)[0][risk_level]

    return int(risk_level), float(risk_score)

# Main app
def main():
    # Custom header
    st.markdown('<h1 class="main-header">🏭 Supply Chain Risk Management</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #6b7280; margin-bottom: 3rem;">Advanced AI-powered risk assessment for supply chain optimization</p>', unsafe_allow_html=True)

    # Load model
    model = load_model()
    if model is None:
        st.error("⚠️ ML model not found! Please run train_model.py first.")
        return

    # Sidebar with enhanced design
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/supply-chain-management.png", width=80)
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio("", ["🏠 Dashboard", "🔮 Risk Prediction", "📋 Risk History", "📈 Analytics", "ℹ️ About"])

    # Quick stats in sidebar
    if page != "🏠 Dashboard":
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Quick Stats")
        df = get_history()
        if not df.empty:
            total_preds = len(df)
            avg_risk = df['risk_score'].mean()
            high_risk_pct = (len(df[df['risk_level'] == 2]) / total_preds * 100) if total_preds > 0 else 0

            st.sidebar.metric("Total Predictions", total_preds)
            st.sidebar.metric("Avg Risk Score", f"{avg_risk:.2f}")
            st.sidebar.metric("High Risk %", f"{high_risk_pct:.1f}%")

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    if page == "🏠 Dashboard":
        show_dashboard_page()
    elif page == "🔮 Risk Prediction":
        show_prediction_page(model)
    elif page == "📋 Risk History":
        show_history_page()
    elif page == "📈 Analytics":
        show_analytics_page()
    elif page == "ℹ️ About":
        show_about_page()

def show_dashboard_page():
    st.markdown('<h2 class="sub-header">📊 Executive Dashboard</h2>', unsafe_allow_html=True)

    df = get_history()

    if df.empty:
        # Welcome message for new users
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            ## Welcome to Supply Chain Risk Management! 🎉

            This dashboard helps you predict and monitor supply chain risks using advanced machine learning.

            ### Key Features:
            - **Risk Prediction**: Assess supplier and product risks in real-time
            - **Historical Analysis**: Track risk trends over time
            - **Advanced Analytics**: Gain insights with comprehensive visualizations
            - **Data Export**: Download reports for further analysis

            ### Getting Started:
            1. Navigate to **Risk Prediction** to make your first assessment
            2. View results and recommendations
            3. Monitor trends in **Analytics**
            """)
        with col2:
            st.image("https://img.icons8.com/fluency/200/000000/analytics.png")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    total_predictions = len(df)
    avg_risk_score = df['risk_score'].mean()
    high_risk_count = len(df[df['risk_level'] == 2])
    latest_risk = df['risk_score'].iloc[0] if not df.empty else 0

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Predictions", f"{total_predictions:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Risk Score", f"{avg_risk_score:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card risk-high">', unsafe_allow_html=True)
        st.metric("High Risk Cases", f"{high_risk_count:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        risk_class = "risk-low" if latest_risk < 0.4 else "risk-medium" if latest_risk < 0.7 else "risk-high"
        st.markdown(f'<div class="metric-card {risk_class}">', unsafe_allow_html=True)
        st.metric("Latest Risk Score", f"{latest_risk:.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Recent activity
    st.markdown("---")
    st.markdown('<h3 class="sub-header">📋 Recent Activity</h3>', unsafe_allow_html=True)

    recent_df = df.head(5)
    for _, row in recent_df.iterrows():
        risk_level = int(row['risk_level'])
        risk_label = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}[risk_level]
        risk_color = {0: "#10b981", 1: "#f59e0b", 2: "#ef4444"}[risk_level]

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"**{row['supplier_id']}** - {row['product_id']}")
        with col2:
            st.write(f"**{risk_label}**")
        with col3:
            st.write(f"Score: {row['risk_score']:.2f}")

        st.markdown("---")

def show_prediction_page(model):
    st.markdown('<h2 class="sub-header">🔮 Risk Assessment Tool</h2>', unsafe_allow_html=True)
    st.markdown("Enter supplier and product details to assess supply chain risk using AI-powered analysis.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.subheader("📋 Assessment Parameters")

        with st.form("prediction_form"):
            # Basic information
            st.markdown("**Supplier & Product Information**")
            col_a, col_b = st.columns(2)
            with col_a:
                supplier_id = st.text_input("Supplier ID", value="SUP001", help="Unique identifier for the supplier")
            with col_b:
                product_id = st.text_input("Product ID", value="PROD001", help="Unique identifier for the product")

            col_c, col_d = st.columns(2)
            with col_c:
                quantity = st.number_input("Order Quantity", min_value=1, value=100, help="Number of units to order")
            with col_d:
                price = st.number_input("Unit Price ($)", min_value=0.01, value=50.0, step=0.01, help="Price per unit")

            delivery_time = st.number_input("Expected Delivery Time (days)", min_value=1, value=7, help="Time from order to delivery")

            st.markdown("---")
            st.markdown("**Risk Factor Assessment (0-1 scale)**")
            st.markdown("*Rate each factor from 0 (no risk) to 1 (maximum risk)*")

            # Risk factors with better descriptions
            supplier_reliability = st.slider("🏢 Supplier Reliability", 0.0, 1.0, 0.8, 0.01,
                                           help="How reliable is this supplier's delivery record?")
            demand_volatility = st.slider("📈 Demand Volatility", 0.0, 1.0, 0.3, 0.01,
                                        help="How variable is the demand for this product?")
            geopolitical_risk = st.slider("🌍 Geopolitical Risk", 0.0, 1.0, 0.2, 0.01,
                                        help="Political or regional instability affecting supply?")
            economic_indicators = st.slider("💰 Economic Indicators", 0.0, 1.0, 0.4, 0.01,
                                          help="Economic conditions affecting supplier performance?")
            historical_delays = st.slider("⏰ Historical Delays", 0.0, 1.0, 0.1, 0.01,
                                        help="Past delivery delays with this supplier/product?")

            st.markdown("---")
            submitted = st.form_submit_button("🚀 Analyze Risk", use_container_width=True)

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

                with st.spinner("🔍 Analyzing risk factors with AI..."):
                    risk_level, risk_score = predict_risk(model, input_data)

                # Save prediction
                save_prediction(input_data, risk_level, risk_score)

                st.success("✅ Risk analysis completed successfully!")
                st.balloons()

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if 'risk_level' in locals() and 'risk_score' in locals():
            st.subheader("📊 Risk Assessment Results")

            # Risk level display with enhanced styling
            risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            risk_classes = {0: "risk-low", 1: "risk-medium", 2: "risk-high"}
            risk_icons = {0: "🟢", 1: "🟡", 2: "🔴"}

            st.markdown(f"""
            <div class="metric-card {risk_classes[risk_level]}" style="text-align: center; margin-bottom: 2rem;">
                <h1 style="margin: 0; font-size: 2.5rem;">{risk_icons[risk_level]}</h1>
                <h2 style="margin: 0.5rem 0;">{risk_labels[risk_level]}</h2>
                <h3 style="margin: 0.5rem 0; opacity: 0.9;">Risk Score: {risk_score:.3f}</h3>
                <p style="margin: 0.5rem 0; opacity: 0.8;">Confidence: {max(risk_score, 0.1):.1%}</p>
            </div>
            """, unsafe_allow_html=True)

            # Risk breakdown with gauge charts
            st.subheader("🔍 Risk Factor Breakdown")
            factors = {
                "Supplier Reliability": supplier_reliability,
                "Demand Volatility": demand_volatility,
                "Geopolitical Risk": geopolitical_risk,
                "Economic Indicators": economic_indicators,
                "Historical Delays": historical_delays
            }

            # Create horizontal bar chart
            factor_df = pd.DataFrame(list(factors.items()), columns=['Factor', 'Score'])
            fig = px.bar(factor_df, x='Score', y='Factor', orientation='h',
                        title="Risk Factor Contributions",
                        color='Score',
                        color_continuous_scale='RdYlGn_r',
                        range_x=[0, 1])
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Recommendations with enhanced styling
            st.subheader("💡 Strategic Recommendations")

            rec_class = {0: "rec-low", 1: "rec-medium", 2: "rec-high"}[risk_level]

            if risk_level == 0:
                st.markdown(f"""
                <div class="recommendation-card {rec_class}">
                    <h4>✅ Low Risk - Proceed with Confidence</h4>
                    <ul>
                        <li>Continue normal procurement processes</li>
                        <li>Maintain standard monitoring protocols</li>
                        <li>Consider volume discounts if applicable</li>
                        <li>Document successful supplier performance</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            elif risk_level == 1:
                st.markdown(f"""
                <div class="recommendation-card {rec_class}">
                    <h4>⚠️ Medium Risk - Implement Monitoring</h4>
                    <ul>
                        <li>Increase supplier performance monitoring</li>
                        <li>Identify and qualify backup suppliers</li>
                        <li>Review and optimize delivery schedules</li>
                        <li>Consider buffer inventory for critical periods</li>
                        <li>Negotiate flexible contract terms</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="recommendation-card {rec_class}">
                    <h4>🚨 High Risk - Immediate Action Required</h4>
                    <ul>
                        <li>Immediately seek alternative suppliers</li>
                        <li>Activate contingency procurement plans</li>
                        <li>Conduct comprehensive contract review</li>
                        <li>Implement emergency inventory buffers</li>
                        <li>Escalate to senior management for approval</li>
                        <li>Consider supply chain diversification</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            # Action buttons
            st.markdown("---")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📋 View Full History", use_container_width=True):
                    st.session_state.page = "📋 Risk History"
                    st.rerun()
            with col_btn2:
                if st.button("📊 View Analytics", use_container_width=True):
                    st.session_state.page = "📈 Analytics"
                    st.rerun()

def show_history_page():
    st.header("📋 Risk History")

    # Get history data
    df = get_history()

    if df.empty:
        st.info("No risk predictions found. Make some predictions first!")
        return

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect("Filter by Risk Level",
                                   options=[0, 1, 2],
                                   default=[0, 1, 2],
                                   format_func=lambda x: {0: "Low", 1: "Medium", 2: "High"}[x])
    with col2:
        supplier_filter = st.text_input("Filter by Supplier ID")
    with col3:
        date_filter = st.date_input("Filter from date", value=None)

    # Apply filters
    filtered_df = df.copy()
    if risk_filter:
        filtered_df = filtered_df[filtered_df['risk_level'].isin(risk_filter)]
    if supplier_filter:
        filtered_df = filtered_df[filtered_df['supplier_id'].str.contains(supplier_filter, case=False)]
    if date_filter:
        filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
        filtered_df = filtered_df[filtered_df['timestamp'].dt.date >= date_filter]

    # Display metrics
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

    # Display table
    st.subheader("Prediction History")
    display_df = filtered_df.copy()
    display_df['risk_level'] = display_df['risk_level'].map({0: "Low", 1: "Medium", 2: "High"})
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    st.dataframe(display_df[['timestamp', 'supplier_id', 'product_id', 'quantity',
                           'price', 'risk_level', 'risk_score']].head(50),
                use_container_width=True)

    # Download option
    if not filtered_df.empty:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download History as CSV",
            data=csv,
            file_name="risk_predictions.csv",
            mime="text/csv"
        )

def show_analytics_page():
    st.markdown('<h2 class="sub-header">📈 Advanced Risk Analytics</h2>', unsafe_allow_html=True)
    st.markdown("Comprehensive insights into supply chain risk patterns and trends.")

    df = get_history()

    if df.empty:
        st.info("📊 No data available for analytics. Make some predictions first!")
        return

    # Key insights
    col1, col2, col3, col4 = st.columns(4)

    total_predictions = len(df)
    avg_risk_score = df['risk_score'].mean()
    risk_trend = df['risk_score'].pct_change().mean() * 100
    high_risk_pct = (len(df[df['risk_level'] == 2]) / total_predictions * 100) if total_predictions > 0 else 0

    with col1:
        st.metric("Total Assessments", f"{total_predictions:,}")
    with col2:
        st.metric("Average Risk Score", f"{avg_risk_score:.2f}")
    with col3:
        trend_icon = "📈" if risk_trend > 0 else "📉"
        st.metric("Risk Trend", f"{trend_icon} {abs(risk_trend):.1f}%")
    with col4:
        st.metric("High Risk Cases", f"{high_risk_pct:.1f}%")

    st.markdown("---")

    # Risk distribution and trends
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Risk Level Distribution")
        risk_counts = df['risk_level'].value_counts().sort_index()
        risk_labels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

        fig = px.pie(values=risk_counts.values,
                    names=[risk_labels[i] for i in risk_counts.index],
                    title="Current Risk Distribution",
                    color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444'],
                    hole=0.4)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Risk Score Evolution")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df_sorted = df.sort_values('timestamp')

        # Add moving average
        df_sorted['risk_ma'] = df_sorted['risk_score'].rolling(window=5, min_periods=1).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sorted['timestamp'], y=df_sorted['risk_score'],
                                mode='markers+lines', name='Risk Score',
                                line=dict(color='#667eea', width=2)))
        fig.add_trace(go.Scatter(x=df_sorted['timestamp'], y=df_sorted['risk_ma'],
                                mode='lines', name='5-Point Moving Average',
                                line=dict(color='#ef4444', width=3, dash='dash')))

        fig.update_layout(title="Risk Score Trend Over Time",
                         xaxis_title="Date", yaxis_title="Risk Score",
                         showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    # Risk factors correlation heatmap
    st.subheader("🔗 Risk Factors Correlation Matrix")
    numeric_cols = ['supplier_reliability', 'demand_volatility', 'geopolitical_risk',
                   'economic_indicators', 'historical_delays', 'risk_score']

    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(corr_matrix,
                   text_auto=True,
                   aspect="auto",
                   title="Correlation Between Risk Factors",
                   color_continuous_scale='RdBu_r',
                   range_color=[-1, 1])
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Supplier performance analysis
    st.subheader("🏢 Supplier Risk Performance")
    supplier_risks = df.groupby('supplier_id')['risk_score'].agg(['mean', 'count', 'std']).round(3)
    supplier_risks = supplier_risks.sort_values('mean', ascending=False).head(15)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(supplier_risks, x=supplier_risks.index, y='mean',
                    title="Top Risky Suppliers (Average Score)",
                    labels={'mean': 'Average Risk Score', 'supplier_id': 'Supplier ID'},
                    color='mean',
                    color_continuous_scale='Reds')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(supplier_risks, x='count', y='mean', size='std',
                        title="Supplier Risk Analysis",
                        labels={'count': 'Number of Assessments', 'mean': 'Average Risk Score'},
                        color='std',
                        color_continuous_scale='Viridis')
        fig.update_traces(text=supplier_risks.index, mode='markers+text', textposition="top center")
        st.plotly_chart(fig, use_container_width=True)

    # Risk factor importance analysis
    st.subheader("🎯 Risk Factor Impact Analysis")
    factor_cols = ['supplier_reliability', 'demand_volatility', 'geopolitical_risk',
                  'economic_indicators', 'historical_delays']

    # Calculate average risk scores by factor ranges
    factor_analysis = []
    for factor in factor_cols:
        low_risk = df[df[factor] <= 0.3]['risk_score'].mean()
        med_risk = df[(df[factor] > 0.3) & (df[factor] <= 0.7)]['risk_score'].mean()
        high_risk = df[df[factor] > 0.7]['risk_score'].mean()

        factor_analysis.append({
            'Factor': factor.replace('_', ' ').title(),
            'Low (0-0.3)': low_risk,
            'Medium (0.3-0.7)': med_risk,
            'High (0.7-1.0)': high_risk
        })

    factor_df = pd.DataFrame(factor_analysis)

    fig = go.Figure()
    for i, row in factor_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row['Low (0-0.3)'], row['Medium (0.3-0.7)'], row['High (0.7-1.0)']],
            theta=['Low Risk Factor', 'Medium Risk Factor', 'High Risk Factor'],
            fill='toself',
            name=row['Factor']
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Risk Factor Impact on Overall Risk Score"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_about_page():
    st.markdown('<h2 class="sub-header">ℹ️ About Supply Chain Risk Management</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ## 🏭 Advanced Supply Chain Risk Assessment Platform

        This comprehensive dashboard leverages machine learning to provide real-time risk assessment for supply chain operations.

        ### 🎯 Key Capabilities

        **Risk Prediction Engine:**
        - AI-powered risk scoring using multiple factors
        - Real-time assessment of supplier and product risks
        - Confidence metrics and detailed factor analysis

        **Comprehensive Analytics:**
        - Historical risk trend analysis
        - Supplier performance monitoring
        - Risk factor correlation insights
        - Advanced visualizations and reporting

        **Data Management:**
        - Persistent storage of all assessments
        - Export capabilities for further analysis
        - Advanced filtering and search options

        ### 🔬 Machine Learning Model

        The system uses a sophisticated classification model trained on historical supply chain data to predict risk levels:

        - **Low Risk**: Business as usual with standard monitoring
        - **Medium Risk**: Enhanced monitoring and contingency planning
        - **High Risk**: Immediate action required with alternative sourcing

        ### 📊 Risk Factors Analyzed

        1. **Supplier Reliability**: Historical delivery performance
        2. **Demand Volatility**: Product demand variability
        3. **Geopolitical Risk**: Regional stability factors
        4. **Economic Indicators**: Market and economic conditions
        5. **Historical Delays**: Past delivery performance issues

        ### 🛠️ Technical Stack

        - **Frontend**: Streamlit with custom CSS styling
        - **Backend**: Python with scikit-learn ML models
        - **Database**: SQLite for data persistence
        - **Visualization**: Plotly for interactive charts
        - **Deployment**: Ready for cloud deployment

        ### 📈 Business Impact

        - **Cost Reduction**: Minimize supply chain disruptions
        - **Risk Mitigation**: Proactive identification of potential issues
        - **Decision Support**: Data-driven procurement decisions
        - **Performance Monitoring**: Track supplier and product performance over time
        """)

    with col2:
        st.markdown("### 🚀 Quick Start Guide")
        st.markdown("""
        1. **Train Model**: Run `train_model.py` first
        2. **Launch Dashboard**: Execute `streamlit run streamlit_app.py`
        3. **Make Assessment**: Navigate to Risk Prediction
        4. **Analyze Results**: Review recommendations and insights
        5. **Monitor Trends**: Use Analytics for ongoing monitoring
        """)

        st.markdown("### 📞 Support")
        st.markdown("""
        For technical support or feature requests:
        - Documentation: [GitHub Wiki]
        - Issues: [GitHub Issues]
        - Email: support@supplychainrisk.com
        """)

        st.markdown("### 📊 System Requirements")
        st.markdown("""
        - Python 3.8+
        - Streamlit
        - scikit-learn
        - pandas
        - plotly
        - SQLite3
        """)

if __name__ == "__main__":
    main()

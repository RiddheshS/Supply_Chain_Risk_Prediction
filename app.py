from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Database setup - using SQLite for easier testing
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///supply_chain_risk.db')
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class RiskData(Base):
    __tablename__ = 'risk_data'
    id = Column(Integer, primary_key=True)
    supplier_id = Column(String)
    product_id = Column(String)
    quantity = Column(Float)
    price = Column(Float)
    delivery_time = Column(Float)
    risk_score = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# Load or train ML model
MODEL_PATH = 'ml_model/risk_prediction_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Sample data for training
    np.random.seed(42)
    n_samples = 1000
    data = {
        'supplier_reliability': np.random.uniform(0, 1, n_samples),
        'demand_volatility': np.random.uniform(0, 1, n_samples),
        'geopolitical_risk': np.random.uniform(0, 1, n_samples),
        'economic_indicators': np.random.uniform(0, 1, n_samples),
        'historical_delays': np.random.uniform(0, 1, n_samples)
    }
    df = pd.DataFrame(data)
    df['risk_level'] = np.random.choice([0, 1, 2], n_samples)  # 0: Low, 1: Medium, 2: High

    X = df.drop('risk_level', axis=1)
    y = df['risk_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, MODEL_PATH)

@app.route('/api/predict_risk', methods=['POST'])
def predict_risk():
    data = request.json
    features = np.array([[
        data['supplier_reliability'],
        data['demand_volatility'],
        data['geopolitical_risk'],
        data['economic_indicators'],
        data['historical_delays']
    ]])
    risk_level = model.predict(features)[0]
    risk_score = model.predict_proba(features)[0][risk_level]

    # Store in database
    session = Session()
    risk_entry = RiskData(
        supplier_id=data.get('supplier_id'),
        product_id=data.get('product_id'),
        quantity=data.get('quantity'),
        price=data.get('price'),
        delivery_time=data.get('delivery_time'),
        risk_score=risk_score
    )
    session.add(risk_entry)
    session.commit()
    session.close()

    return jsonify({
        'risk_level': int(risk_level),
        'risk_score': float(risk_score),
        'message': 'Risk prediction completed'
    })

@app.route('/api/risk_history', methods=['GET'])
def get_risk_history():
    session = Session()
    risks = session.query(RiskData).all()
    session.close()

    result = [{
        'id': r.id,
        'supplier_id': r.supplier_id,
        'product_id': r.product_id,
        'quantity': r.quantity,
        'price': r.price,
        'delivery_time': r.delivery_time,
        'risk_score': r.risk_score,
        'timestamp': r.timestamp.isoformat()
    } for r in risks]

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

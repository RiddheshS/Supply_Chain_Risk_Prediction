import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def generate_sample_data(n_samples=10000):
    np.random.seed(42)
    data = {
        'supplier_reliability': np.random.uniform(0, 1, n_samples),
        'demand_volatility': np.random.uniform(0, 1, n_samples),
        'geopolitical_risk': np.random.uniform(0, 1, n_samples),
        'economic_indicators': np.random.uniform(0, 1, n_samples),
        'historical_delays': np.random.uniform(0, 1, n_samples),
        'supplier_diversity': np.random.uniform(0, 1, n_samples),
        'inventory_levels': np.random.uniform(0, 1, n_samples),
        'transportation_costs': np.random.uniform(0, 1, n_samples)
    }
    df = pd.DataFrame(data)

    # Create risk level based on weighted factors
    risk_factors = (
        0.3 * data['supplier_reliability'] +
        0.2 * data['demand_volatility'] +
        0.25 * data['geopolitical_risk'] +
        0.15 * data['economic_indicators'] +
        0.1 * data['historical_delays']
    )

    df['risk_level'] = pd.cut(risk_factors, bins=[0, 0.3, 0.7, 1], labels=[0, 1, 2])  # 0: Low, 1: Medium, 2: High
    df['risk_level'] = df['risk_level'].astype(int)

    return df

def train_model():
    print("Generating sample data...")
    df = generate_sample_data()

    print("Preparing features and target...")
    # Use only the features collected in the UI for consistency
    feature_columns = ['supplier_reliability', 'demand_volatility', 'geopolitical_risk',
                      'economic_indicators', 'historical_delays']
    X = df[feature_columns]
    y = df['risk_level']

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Saving model...")
    os.makedirs('ml_model', exist_ok=True)
    joblib.dump(model, 'ml_model/risk_prediction_model.pkl')
    print("Model saved successfully!")

    # Save sample data for reference
    df.to_csv('data/sample_supply_chain_data.csv', index=False)
    print("Sample data saved to data/sample_supply_chain_data.csv")

if __name__ == '__main__':
    train_model()

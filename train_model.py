import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def _make_synthetic_dataset(n=2000, random_state=42):
    rng = np.random.default_rng(random_state)

    # Features in [0,1] as expected by the UI sliders
    supplier_reliability = rng.uniform(0, 1, n)
    demand_volatility    = rng.uniform(0, 1, n)
    geopolitical_risk    = rng.uniform(0, 1, n)
    economic_indicators  = rng.uniform(0, 1, n)
    historical_delays    = rng.uniform(0, 1, n)

    # Build a synthetic "risk score" with some noise
    # Lower reliability + higher volatility/risk/delays -> higher risk
    base = (
        (1 - supplier_reliability) * 0.35
        + demand_volatility * 0.20
        + geopolitical_risk * 0.20
        + (1 - economic_indicators) * 0.15
        + historical_delays * 0.10
    )
    noise = rng.normal(0, 0.05, n)
    risk_cont = np.clip(base + noise, 0, 1)

    # Discretize into 3 classes: 0=Low, 1=Medium, 2=High
    # thresholds chosen to get a balanced-ish dataset
    y = np.digitize(risk_cont, bins=[0.33, 0.66])

    df = pd.DataFrame({
        "supplier_reliability": supplier_reliability,
        "demand_volatility": demand_volatility,
        "geopolitical_risk": geopolitical_risk,
        "economic_indicators": economic_indicators,
        "historical_delays": historical_delays,
        "risk_level": y
    })
    return df

def train_model():
    print("Generating synthetic training data...")
    df = _make_synthetic_dataset()

    X = df[["supplier_reliability", "demand_volatility", "geopolitical_risk",
            "economic_indicators", "historical_delays"]]
    y = df["risk_level"]

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Saving model...")
    os.makedirs("ml_model", exist_ok=True)
    joblib.dump(model, "ml_model/risk_prediction_model.pkl")
    print("Model saved to ml_model/risk_prediction_model.pkl")

    print("Saving sample data for the app...")
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sample_supply_chain_data.csv", index=False)
    print("Sample data saved to data/sample_supply_chain_data.csv")

if __name__ == "__main__":
    train_model()

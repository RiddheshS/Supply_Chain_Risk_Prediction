import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_supply_chain_data(n_records=1000):
    np.random.seed(42)

    # Generate supplier and product IDs
    suppliers = [f'SUP{i:03d}' for i in range(1, 51)]  # 50 suppliers
    products = [f'PROD{i:03d}' for i in range(1, 101)]  # 100 products

    data = []

    for _ in range(n_records):
        supplier = np.random.choice(suppliers)
        product = np.random.choice(products)

        # Generate realistic supply chain data
        quantity = np.random.uniform(10, 1000)
        price = np.random.uniform(5, 500)
        delivery_time = np.random.uniform(1, 30)  # days

        # Risk factors (0-1 scale)
        supplier_reliability = np.random.beta(2, 2)  # Beta distribution for realistic spread
        demand_volatility = np.random.beta(2, 2)
        geopolitical_risk = np.random.beta(2, 2)
        economic_indicators = np.random.beta(2, 2)
        historical_delays = np.random.beta(2, 2)

        # Calculate risk score based on factors
        risk_score = (
            0.3 * supplier_reliability +
            0.2 * demand_volatility +
            0.25 * geopolitical_risk +
            0.15 * economic_indicators +
            0.1 * historical_delays
        )

        # Determine risk level
        if risk_score < 0.3:
            risk_level = 'Low'
        elif risk_score < 0.7:
            risk_level = 'Medium'
        else:
            risk_level = 'High'

        # Generate timestamp
        days_ago = np.random.randint(0, 365)
        timestamp = datetime.now() - timedelta(days=days_ago)

        data.append({
            'supplier_id': supplier,
            'product_id': product,
            'quantity': round(quantity, 2),
            'price': round(price, 2),
            'delivery_time': round(delivery_time, 2),
            'supplier_reliability': round(supplier_reliability, 4),
            'demand_volatility': round(demand_volatility, 4),
            'geopolitical_risk': round(geopolitical_risk, 4),
            'economic_indicators': round(economic_indicators, 4),
            'historical_delays': round(historical_delays, 4),
            'risk_score': round(risk_score, 4),
            'risk_level': risk_level,
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S')
        })

    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    print("Generating sample supply chain data...")
    df = generate_supply_chain_data(5000)  # Generate 5000 records
    df.to_csv('data/sample_supply_chain_data.csv', index=False)
    print("Sample data saved to data/sample_supply_chain_data.csv")
    print(f"Generated {len(df)} records")
    print("\nData summary:")
    print(df.describe())
    print("\nRisk level distribution:")
    print(df['risk_level'].value_counts())

# Supply Chain Risk Prediction

A comprehensive web application for predicting and managing supply chain risks using machine learning.

## Features

- **Risk Prediction**: Predict supply chain risks based on multiple factors including supplier reliability, demand volatility, geopolitical risks, and economic indicators.
- **Interactive Dashboard**: User-friendly web interface for inputting data and visualizing risk predictions.
- **Historical Data Tracking**: Store and view historical risk assessments.
- **Machine Learning Model**: Random Forest classifier trained on supply chain data.
- **Database Integration**: PostgreSQL database for persistent data storage.

## Technology Stack

### Backend
- **Flask**: Python web framework
- **SQLAlchemy**: ORM for database operations
- **scikit-learn**: Machine learning library
- **pandas & numpy**: Data processing
- **PostgreSQL**: Relational database

### Frontend
- **React**: JavaScript library for building user interfaces
- **Material-UI**: React components library
- **Chart.js**: Data visualization
- **Axios**: HTTP client for API calls

### Machine Learning
- **Random Forest Classifier**: For risk prediction
- **Joblib**: Model serialization

## Installation

### Prerequisites
- Python 3.8+
- Node.js 14+
- PostgreSQL

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd supply_chain_risk_prediction/backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up the database:
   - Create a PostgreSQL database named `supply_chain_risk`
   - Update the `DATABASE_URL` in `app.py` with your database credentials

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd supply_chain_risk_prediction/frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

### Machine Learning Model
1. Navigate to the ml_model directory:
   ```bash
   cd supply_chain_risk_prediction/ml_model
   ```

2. Train the model:
   ```bash
   python train_model.py
   ```

## Usage

### Running the Application
1. Start the backend server:
   ```bash
   cd supply_chain_risk_prediction/backend
   python app.py
   ```
   The backend will run on http://localhost:5000

2. Start the frontend:
   ```bash
   cd supply_chain_risk_prediction/frontend
   npm start
   ```
   The frontend will run on http://localhost:3000

### API Endpoints
- `POST /api/predict_risk`: Predict risk for given input data
- `GET /api/risk_history`: Retrieve historical risk assessments

## Data Structure

### Risk Prediction Input
- supplier_id: Unique supplier identifier
- product_id: Unique product identifier
- quantity: Order quantity
- price: Unit price
- delivery_time: Expected delivery time in days
- supplier_reliability: Reliability score (0-1)
- demand_volatility: Demand volatility score (0-1)
- geopolitical_risk: Geopolitical risk score (0-1)
- economic_indicators: Economic indicators score (0-1)
- historical_delays: Historical delays score (0-1)

### Risk Levels
- 0: Low Risk
- 1: Medium Risk
- 2: High Risk

## Sample Data

Generate sample data using:
```bash
cd supply_chain_risk_prediction/data
python generate_sample_data.py
```

This will create `sample_supply_chain_data.csv` with 5000 sample records.

## Model Training

The machine learning model is trained using synthetic data that simulates real-world supply chain scenarios. The model considers multiple risk factors and outputs a risk level and probability score.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

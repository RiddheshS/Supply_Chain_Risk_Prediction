#!/bin/sh
set -e

cd /app

# Ensure data dir exists (some trainers write sample CSVs)
mkdir -p data

# Train once if model is missing
if [ ! -f "ml_model/risk_prediction_model.pkl" ]; then
  echo "Model not found. Training now..."
  python -c "import ml_model.train_model as t; t.train_model()"
else
  echo "Model found. Skipping training."
fi

# Start Streamlit
exec streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0

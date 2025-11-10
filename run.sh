#!/bin/bash

# Supply Chain Risk Prediction - Run Script

echo "Starting Supply Chain Risk Prediction Application..."

# Check if Docker is available
if command -v docker-compose &> /dev/null; then
    echo "Using Docker Compose to run the application..."
    docker-compose up --build
else
    echo "Docker Compose not found. Running locally..."

    # Check if Python virtual environment exists
    if [ ! -d "backend/venv" ]; then
        echo "Creating Python virtual environment..."
        cd backend
        python -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        cd ..
    else
        echo "Activating existing virtual environment..."
        source backend/venv/bin/activate
    fi

    # Train the model if not exists
    if [ ! -f "ml_model/risk_prediction_model.pkl" ]; then
        echo "Training machine learning model..."
        cd ml_model
        python train_model.py
        cd ..
    fi

    # Generate sample data if not exists
    if [ ! -f "data/sample_supply_chain_data.csv" ]; then
        echo "Generating sample data..."
        cd data
        python generate_sample_data.py
        cd ..
    fi

    # Start backend in background
    echo "Starting backend server..."
    cd backend
    python app.py &
    BACKEND_PID=$!
    cd ..

    # Start frontend
    echo "Starting frontend..."
    cd frontend
    npm install
    npm start &
    FRONTEND_PID=$!
    cd ..

    echo "Application started successfully!"
    echo "Backend: http://localhost:5000"
    echo "Frontend: http://localhost:3000"
    echo ""
    echo "Press Ctrl+C to stop the application"

    # Wait for user interrupt
    trap "echo 'Stopping application...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
    wait
fi

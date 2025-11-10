# Use consistent casing
FROM python:3.11-slim AS app

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps (useful for building common wheels)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python libs directly (no requirements.txt)
# Add more here if your code imports others (e.g., xgboost, lightgbm, opencv-python)
RUN pip install --no-cache-dir \
    streamlit pandas numpy scikit-learn joblib plotly

# Copy project files
COPY . /app

# Ensure start.sh has Unix line endings and is executable
# (prevents /bin/sh^M errors if file was created on Windows)
RUN sed -i 's/\r$//' /app/start.sh && chmod +x /app/start.sh

# Copy Streamlit config into container so it always binds correctly
RUN mkdir -p /root/.streamlit && \
    cp /app/.streamlit/config.toml /root/.streamlit/config.toml && \
    chmod 600 /root/.streamlit/config.toml

# Belt-and-suspenders: also set env vars
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

# Streamlit port
EXPOSE 8501

# Start script: ensures data/, trains model if missing, then runs Streamlit
CMD ["sh", "/app/start.sh"]

# python:3.10-slim always resolves to the latest 3.10.x patch (currently 3.10.16)
# DO NOT use 3.10.20 — that tag does not exist on Docker Hub
FROM python:3.10-slim

WORKDIR /app

# System deps needed by curl_cffi and matplotlib
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy rest of the app
COPY . .

EXPOSE 8501

# $PORT is set by Render automatically; fall back to 8501 locally
# XSRF and CORS protection are left at Streamlit's secure defaults (enabled).
CMD streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true

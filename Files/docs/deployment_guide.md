# Macroeconomic Inflation Hedge Analytics Platform
## Deployment Guide

### Version 2.0.0 Enterprise
### Author: DevOps Engineering Team
### Date: August 2025

---

## Overview

This deployment guide covers the Macroeconomic Inflation Hedge Analytics Platform deployment with focus on economic data processing, time series analysis, and real-time economic monitoring.

## Prerequisites

### System Requirements

**Production Requirements**:
- CPU: 16 cores (Intel Xeon recommended)
- RAM: 64GB DDR4 ECC memory
- Storage: 1TB NVMe SSD
- Network: 1Gbps connectivity

### Software Dependencies

- **Runtime**: Python 3.11+, R 4.0+ (for econometric models)
- **Databases**: PostgreSQL 15+, InfluxDB 2.0+, Redis 7+
- **Economic Libraries**: Statsmodels, PyMC, Prophet, Arch

## Local Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/enterprise/inflation-hedge-analytics.git
cd inflation-hedge-analytics

# Create virtual environment
python3.11 -m venv inflation_env
source inflation_env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-econometric.txt
```

### 2. Economic Database Setup

```bash
# InfluxDB for time series economic data
docker run -p 8086:8086 -v influxdb:/var/lib/influxdb2 influxdb:2.7

# PostgreSQL for economic metadata
sudo apt install postgresql-15
sudo systemctl start postgresql

# Create economic database
sudo -u postgres createdb economic_intelligence
```

### 3. Environment Configuration

```bash
# Create .env file
cat > .env << EOF
# Database Configuration
DATABASE_URL=postgresql://economics:password@localhost/economic_intelligence
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=your-influxdb-token
REDIS_URL=redis://localhost:6379

# Economic Data APIs
FRED_API_KEY=your-fred-api-key
OECD_API_KEY=your-oecd-api-key
BLOOMBERG_API_KEY=your-bloomberg-key
QUANDL_API_KEY=your-quandl-key

# Model Configuration
MODEL_PATH=./models
ECONOMIC_DATA_PATH=./data/economic
FORECAST_HORIZON_MAX=60
UPDATE_FREQUENCY=DAILY

# Security
JWT_SECRET_KEY=your-jwt-secret-32-characters
ENCRYPTION_KEY=your-encryption-key-32-chars

# Monitoring
PROMETHEUS_ENABLED=true
LOG_LEVEL=INFO
EOF
```

### 4. Start Development Environment

```bash
# Start economic data ingestion
python scripts/start_data_ingestion.py

# Run application
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Verify system health
curl http://localhost:8000/health
curl http://localhost:8000/economic/status
```

## Docker Deployment

### 1. Economic Analytics Dockerfile

```dockerfile
FROM python:3.11-slim AS economic-base

# Install system dependencies for economic libraries
RUN apt-get update && apt-get install -y \
    gcc g++ gfortran \
    libblas-dev liblapack-dev \
    r-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-econometric.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-econometric.txt

# Install R packages for econometric models
RUN R -e "install.packages(c('forecast', 'vars', 'urca'), repos='http://cran.rstudio.com/')"

FROM economic-base AS production

RUN groupadd -r economics && useradd -r -g economics economics

COPY --chown=economics:economics Files/ ./Files/
COPY --chown=economics:economics models/ ./models/
COPY --chown=economics:economics *.py ./

ENV PYTHONPATH=/app
USER economics

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

### 2. Docker Compose for Economic Platform

```yaml
version: '3.8'

services:
  economics-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://economics:password@postgres:5432/economic_intelligence
      - INFLUXDB_URL=http://influxdb:8086
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    depends_on:
      - postgres
      - influxdb
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: economic_intelligence
      POSTGRES_USER: economics
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  influxdb:
    image: influxdb:2.7-alpine
    environment:
      DOCKER_INFLUXDB_INIT_MODE: setup
      DOCKER_INFLUXDB_INIT_USERNAME: admin
      DOCKER_INFLUXDB_INIT_PASSWORD: SecurePassword123
      DOCKER_INFLUXDB_INIT_ORG: economics-enterprise
      DOCKER_INFLUXDB_INIT_BUCKET: economic-data
    volumes:
      - influxdb_data:/var/lib/influxdb2
    ports:
      - "8086:8086"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 2gb
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  postgres_data:
  influxdb_data:
  redis_data:
```

## Kubernetes Deployment

### 1. Economic Application Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: economics-app
  namespace: economic-intelligence
spec:
  replicas: 2
  selector:
    matchLabels:
      app: economics-app
  template:
    metadata:
      labels:
        app: economics-app
    spec:
      containers:
      - name: economics-app
        image: enterprise/economic-intelligence:2.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: economics-secrets
              key: database-url
        - name: FRED_API_KEY
          valueFrom:
            secretKeyRef:
              name: economics-secrets
              key: fred-api-key
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
          limits:
            cpu: "4"
            memory: "16Gi"
        volumeMounts:
        - name: economic-models
          mountPath: /app/models
        - name: economic-data
          mountPath: /app/data
      volumes:
      - name: economic-models
        persistentVolumeClaim:
          claimName: models-pvc
      - name: economic-data
        persistentVolumeClaim:
          claimName: data-pvc
```

## Performance Optimization

### 1. Economic Model Optimization

```python
# Optimized econometric model configuration
ECONOMETRIC_CONFIG = {
    'parallel_processing': True,
    'num_cores': 4,
    'memory_efficient': True,
    'cache_forecasts': True,
    'batch_size': 1000
}
```

### 2. Time Series Database Tuning

```bash
# InfluxDB optimization for economic data
echo "max-series-per-database = 10000000" >> /etc/influxdb/influxdb.conf
echo "max-values-per-tag = 1000000" >> /etc/influxdb/influxdb.conf
systemctl restart influxdb
```

## Monitoring Economic Models

### 1. Economic Metrics

```yaml
# Prometheus economic metrics
- name: inflation_forecast_accuracy
  help: Accuracy of inflation forecasts
  type: gauge
  
- name: economic_data_freshness
  help: Time since last economic data update
  type: gauge
  
- name: model_performance
  help: Economic model performance metrics
  type: histogram
```

### 2. Economic Alerts

```yaml
groups:
- name: economic-alerts
  rules:
  - alert: InflationForecastAccuracyDrop
    expr: inflation_forecast_accuracy < 0.85
    for: 1h
    labels:
      severity: warning
    annotations:
      summary: "Inflation forecast accuracy below threshold"
      
  - alert: EconomicDataStale
    expr: economic_data_freshness > 3600
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Economic data not updated in over 1 hour"
```

This deployment guide provides comprehensive instructions for deploying the Macroeconomic Inflation Hedge Analytics Platform with specialized focus on economic data processing and time series analysis requirements.
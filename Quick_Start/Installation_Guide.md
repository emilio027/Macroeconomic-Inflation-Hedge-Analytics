# 🚀 Quick Start Installation Guide
## Macroeconomic Inflation Hedge Analytics Platform

> **Get inflation protection analytics running in under 10 minutes**

---

## 📋 **Prerequisites**

### **Minimum Requirements:**
- 💻 **Python 3.9+** 
- 🐳 **Docker** (recommended)
- 💾 **8GB RAM** minimum
- 📊 **Economic Data APIs** (optional for live data)

---

## ⚡ **Quick Docker Setup (Recommended)**

### **1. Clone and Setup**
```bash
# Clone the repository
git clone <repository-url>
cd Macroeconomic-Inflation-Hedge-Analytics

# Copy environment template
cp Resources/configuration/.env.example .env
```

### **2. Configure Environment**
Edit `.env` file:
```bash
# Economic Data APIs
FRED_API_KEY=your_fred_api_key
ECONOMIC_INDICATORS_API=your_api_key

# Portfolio Configuration
BASE_CURRENCY=USD
PORTFOLIO_SIZE=1000000
RISK_TOLERANCE=moderate
```

### **3. Launch Platform**
```bash
# Start all services
docker-compose up -d

# Access the platform
# Web Interface: http://localhost:8080
# API Docs: http://localhost:8080/api/docs
```

---

## ✅ **Verification Steps**

### **Test Economic Forecasting**
```bash
# Test inflation prediction
curl -X POST http://localhost:8080/api/inflation-forecast \
  -H "Content-Type: application/json" \
  -d '{"horizon_months": 12, "confidence_level": 0.95}'
```

### **Verify Portfolio Optimization**
```bash
# Test hedge strategy
curl -X POST http://localhost:8080/api/hedge-optimization \
  -H "Content-Type: application/json" \
  -d '{"portfolio_value": 1000000, "inflation_target": 0.03}'
```

---

## 🎯 **Next Steps**

1. **📊 Interactive Demo**: Open `interactive_demo.html`
2. **📈 Economic Analysis**: Access dashboard at localhost:8080
3. **🔗 Portfolio Optimization**: Configure your hedge strategies

---

**🎉 Your Macroeconomic Intelligence Platform is ready!**
# Macroeconomic Inflation Hedge Analytics Platform
## API Documentation

### Version 2.0.0 Enterprise
### Author: API Documentation Team
### Date: August 2025

---

## Overview

The Macroeconomic Inflation Hedge Analytics Platform provides comprehensive APIs for inflation forecasting, economic scenario analysis, and hedging strategy optimization.

**Base URL**: `https://api.inflation.enterprise.com/v2`
**Authentication**: Bearer Token (OAuth 2.0)
**Rate Limiting**: 2,000 requests/minute per API key

## Core Economic APIs

### 1. Inflation Forecasting

#### Get Inflation Forecast

**Endpoint**: `POST /inflation/forecast`

**Request Body**:
```json
{
  "country": "US",
  "forecast_horizon": ["1M", "3M", "6M", "1Y", "2Y"],
  "confidence_intervals": [0.68, 0.95],
  "inflation_type": "CORE_CPI",
  "conditioning_data": {
    "oil_price": 85.0,
    "unemployment_rate": 3.7,
    "fed_funds_rate": 5.25
  }
}
```

**Response**:
```json
{
  "forecast_id": "FORE-2025-08-001",
  "country": "US",
  "base_date": "2025-08-18",
  "forecasts": [
    {
      "horizon": "1M",
      "forecast": 0.023,
      "confidence_intervals": {
        "68": [0.019, 0.027],
        "95": [0.012, 0.034]
      },
      "regime_probability": {
        "low_inflation": 0.23,
        "moderate_inflation": 0.67,
        "high_inflation": 0.10
      }
    }
  ],
  "model_confidence": 0.942,
  "key_drivers": [
    {
      "factor": "Labor_Market",
      "contribution": 0.34,
      "direction": "INFLATIONARY"
    }
  ]
}
```

### 2. Hedging Strategy Optimization

#### Optimize Hedge Portfolio

**Endpoint**: `POST /hedging/optimize`

**Request Body**:
```json
{
  "portfolio_id": "PORT-HEDGE-001",
  "inflation_target": 0.02,
  "risk_tolerance": "MODERATE",
  "time_horizon": 24,
  "constraints": {
    "max_tips_allocation": 0.40,
    "max_commodity_allocation": 0.25,
    "min_liquidity": 0.15
  },
  "economic_scenario": "BASE_CASE"
}
```

**Response**:
```json
{
  "optimization_id": "OPT-2025-08-001",
  "optimal_allocation": {
    "TIPS": 0.35,
    "Commodities": 0.20,
    "Real_Estate": 0.18,
    "Equities": 0.15,
    "Cash": 0.12
  },
  "hedge_effectiveness": 0.873,
  "expected_inflation_beta": 0.92,
  "tracking_error": 0.034,
  "cost_of_hedging": 0.0067
}
```

### 3. Economic Scenario Analysis

#### Generate Economic Scenarios

**Endpoint**: `POST /scenarios/generate`

**Request Body**:
```json
{
  "scenario_type": "INFLATION_STRESS",
  "num_scenarios": 1000,
  "time_horizon": 60,
  "shock_parameters": {
    "oil_shock": 2.0,
    "supply_chain_disruption": 1.5,
    "monetary_policy_error": 1.0
  }
}
```

**Response**:
```json
{
  "scenario_id": "SCEN-2025-08-001",
  "scenarios": [
    {
      "scenario_id": 1,
      "inflation_path": [0.025, 0.031, 0.045],
      "gdp_growth": [0.021, 0.018, 0.012],
      "interest_rates": [0.0525, 0.0575, 0.0625],
      "probability": 0.023
    }
  ],
  "summary_statistics": {
    "mean_inflation": 0.034,
    "p95_inflation": 0.067,
    "recession_probability": 0.23
  }
}
```

## Data Models

### Economic Forecast Schema

```json
{
  "type": "object",
  "properties": {
    "country": {"type": "string", "enum": ["US", "EU", "UK", "JP", "CA"]},
    "forecast_horizon": {"type": "array", "items": {"type": "string"}},
    "inflation_type": {"type": "string", "enum": ["HEADLINE_CPI", "CORE_CPI", "PCE"]},
    "confidence_intervals": {"type": "array", "items": {"type": "number"}}
  }
}
```

This API documentation provides essential endpoints for accessing the platform's inflation forecasting and hedging optimization capabilities.
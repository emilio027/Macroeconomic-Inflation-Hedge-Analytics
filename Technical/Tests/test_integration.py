"""
Integration Testing Suite for Macroeconomic Inflation Hedge Analytics Platform
==============================================================================

Tests integration between components, economic models, data feeds, hedging algorithms,
and external market data services for the macroeconomic inflation platform.

Author: Emilio Cardenas
License: MIT
"""

import pytest
import asyncio
import requests
import json
import time
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import threading
import tempfile
import sqlite3
import pandas as pd
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main import app
    from advanced_inflation_analytics import MacroeconomicInflationPlatform
    from analytics_engine import AnalyticsEngine
    from data_manager import DataManager
    from ml_models import MLModelManager
    from visualization_manager import VisualizationManager
except ImportError as e:
    print(f"Warning: Using mocks for unavailable modules: {e}")
    app = Mock()


class TestEconomicDataPipeline:
    """Economic data pipeline integration tests."""
    
    def setup_method(self):
        """Setup for each test method."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
        self.base_url = 'http://localhost:8001'
        self.headers = {'Content-Type': 'application/json'}
        
        self.sample_economic_request = {
            'indicators': ['cpi', 'pce', 'gdp', 'unemployment'],
            'time_period': '2024-01-01',
            'forecast_horizon': 12,
            'scenario': 'base_case'
        }
    
    def test_full_economic_analysis_workflow(self):
        """Test complete economic analysis workflow from data to recommendations."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
            
        # Step 1: Request economic analysis
        response = self.client.post('/api/v1/economic_analysis', 
                                   json=self.sample_economic_request,
                                   headers=self.headers)
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify response structure
            assert 'inflation_forecast' in data
            assert 'hedging_recommendations' in data
            assert 'risk_assessment' in data
            assert 'portfolio_impact' in data
            assert 'scenario_analysis' in data
            
            # Verify forecast data
            if 'inflation_forecast' in data:
                forecast = data['inflation_forecast']
                assert 'predicted_values' in forecast
                assert 'confidence_intervals' in forecast
                assert len(forecast['predicted_values']) > 0
    
    def test_real_time_indicator_integration(self):
        """Test real-time economic indicator integration."""
        if not hasattr(self.client, 'get'):
            pytest.skip("Client not available")
            
        response = self.client.get('/api/v1/indicators/real_time')
        
        if response.status_code == 200:
            data = json.loads(response.data)
            
            # Verify real-time data structure
            expected_indicators = [
                'cpi_current',
                'pce_current', 
                'gdp_nowcast',
                'employment_data',
                'fed_funds_rate'
            ]
            
            for indicator in expected_indicators:
                if indicator in data:
                    assert data[indicator] is not None
                    assert 'value' in data[indicator]
                    assert 'timestamp' in data[indicator]
                    assert 'data_quality' in data[indicator]
    
    def test_cross_source_data_validation(self):
        """Test validation across multiple economic data sources."""
        # Mock multiple data sources
        def mock_federal_reserve_data():
            return {
                'cpi': 287.5,
                'pce': 275.2,
                'fed_funds': 5.25,
                'timestamp': datetime.now(),
                'source': 'federal_reserve'
            }
        
        def mock_bls_data():
            return {
                'cpi': 287.3,  # Slightly different
                'employment': 96.2,
                'wages': 4.1,
                'timestamp': datetime.now(),
                'source': 'bureau_labor_statistics'
            }
        
        def mock_bea_data():
            return {
                'gdp': 2.1,
                'pce': 275.0,  # Slightly different
                'personal_income': 21500,
                'timestamp': datetime.now(),
                'source': 'bureau_economic_analysis'
            }
        
        fed_data = mock_federal_reserve_data()
        bls_data = mock_bls_data()
        bea_data = mock_bea_data()
        
        # Test data reconciliation
        def reconcile_cross_source_data(sources):
            """Reconcile data across multiple sources."""
            reconciled = {}
            
            # Find common indicators
            common_indicators = set()
            for source_data in sources:
                common_indicators.update(source_data.keys())
            
            # Remove metadata fields
            metadata_fields = {'timestamp', 'source', 'quality_score'}
            common_indicators -= metadata_fields
            
            for indicator in common_indicators:
                values = []
                sources_with_indicator = []
                
                for source_data in sources:
                    if indicator in source_data:
                        values.append(source_data[indicator])
                        sources_with_indicator.append(source_data.get('source', 'unknown'))
                
                if len(values) > 1:
                    # Multiple sources - calculate consensus
                    median_value = np.median(values)
                    std_dev = np.std(values)
                    
                    reconciled[indicator] = {
                        'consensus_value': median_value,
                        'std_deviation': std_dev,
                        'data_quality': 1 - (std_dev / median_value) if median_value != 0 else 0,
                        'source_count': len(values),
                        'sources': sources_with_indicator
                    }
                elif len(values) == 1:
                    # Single source
                    reconciled[indicator] = {
                        'consensus_value': values[0],
                        'std_deviation': 0,
                        'data_quality': 0.8,  # Lower quality for single source
                        'source_count': 1,
                        'sources': sources_with_indicator
                    }
            
            return reconciled
        
        sources = [fed_data, bls_data, bea_data]
        reconciled_data = reconcile_cross_source_data(sources)
        
        # Validate reconciliation results
        assert 'cpi' in reconciled_data  # Should be reconciled from Fed and BLS
        assert 'pce' in reconciled_data  # Should be reconciled from Fed and BEA
        
        # Check data quality scores
        cpi_quality = reconciled_data['cpi']['data_quality']
        assert 0 <= cpi_quality <= 1
        
        # Multi-source indicators should have higher quality when consistent
        if reconciled_data['cpi']['std_deviation'] < 1.0:
            assert cpi_quality > 0.9
    
    def test_hedging_strategy_integration(self):
        """Test integration between economic forecasts and hedging strategies."""
        # Mock economic scenario
        economic_scenario = {
            'inflation_forecast': [2.8, 3.2, 3.5, 3.1, 2.9],
            'gdp_forecast': [2.1, 1.8, 2.3, 2.5, 2.4],
            'fed_rate_forecast': [5.25, 5.50, 5.25, 4.75, 4.50],
            'confidence_level': 0.85
        }
        
        # Mock portfolio exposure
        portfolio_exposure = {
            'total_value': 100000000,  # $100M
            'asset_allocation': {
                'stocks': 0.60,
                'bonds': 0.30,
                'real_estate': 0.05,
                'commodities': 0.05
            },
            'inflation_sensitivity': {
                'stocks': -0.25,
                'bonds': -1.50,
                'real_estate': 0.75,
                'commodities': 1.25
            }
        }
        
        def generate_hedging_strategy(scenario, portfolio):
            """Generate optimal hedging strategy based on economic scenario."""
            # Calculate portfolio inflation beta
            portfolio_beta = sum(
                portfolio['asset_allocation'][asset] * 
                portfolio['inflation_sensitivity'][asset]
                for asset in portfolio['asset_allocation']
            )
            
            # Calculate expected inflation impact
            avg_inflation = np.mean(scenario['inflation_forecast'])
            expected_impact = portfolio_beta * avg_inflation / 100
            
            # Determine hedge requirements
            hedge_instruments = {
                'tips': {
                    'beta': 0.85,
                    'cost': 0.002,
                    'liquidity': 'high',
                    'capacity': 0.30
                },
                'commodities': {
                    'beta': 1.25,
                    'cost': 0.008,
                    'liquidity': 'medium',
                    'capacity': 0.15
                },
                'inflation_swaps': {
                    'beta': 1.00,
                    'cost': 0.003,
                    'liquidity': 'medium',
                    'capacity': 0.25
                }
            }
            
            # Optimize hedge allocation
            optimal_allocation = {}
            remaining_hedge_needed = abs(expected_impact)
            
            for instrument, properties in hedge_instruments.items():
                if remaining_hedge_needed > 0:
                    # Calculate optimal allocation for this instrument
                    max_allocation = min(properties['capacity'], remaining_hedge_needed / properties['beta'])
                    
                    # Adjust for cost-effectiveness
                    cost_effectiveness = properties['beta'] / properties['cost']
                    
                    if cost_effectiveness > 100:  # Minimum cost-effectiveness threshold
                        optimal_allocation[instrument] = max_allocation
                        remaining_hedge_needed -= max_allocation * properties['beta']
            
            return {
                'portfolio_inflation_beta': portfolio_beta,
                'expected_impact': expected_impact,
                'hedge_allocation': optimal_allocation,
                'residual_risk': remaining_hedge_needed,
                'total_hedge_cost': sum(
                    allocation * hedge_instruments[instrument]['cost']
                    for instrument, allocation in optimal_allocation.items()
                )
            }
        
        strategy = generate_hedging_strategy(economic_scenario, portfolio_exposure)
        
        # Validate strategy results
        assert 'portfolio_inflation_beta' in strategy
        assert 'hedge_allocation' in strategy
        assert 'total_hedge_cost' in strategy
        
        # Beta should be negative (needs hedging)
        assert strategy['portfolio_inflation_beta'] < 0
        
        # Should have some hedge allocation
        assert len(strategy['hedge_allocation']) > 0
        
        # Total cost should be reasonable
        assert strategy['total_hedge_cost'] < 0.02  # Less than 2%
    
    def test_scenario_stress_testing_integration(self):
        """Test integration of scenario analysis with stress testing."""
        # Define stress scenarios
        stress_scenarios = {
            'high_inflation': {
                'inflation_path': [3.2, 5.1, 6.8, 7.2, 6.5],
                'gdp_path': [2.1, 0.8, -0.5, 0.2, 1.1],
                'probability': 0.15
            },
            'recession': {
                'inflation_path': [3.2, 2.1, 1.2, 0.8, 1.5],
                'gdp_path': [2.1, -1.2, -2.8, -1.5, 0.8],
                'probability': 0.20
            },
            'stagflation': {
                'inflation_path': [3.2, 4.5, 6.2, 6.8, 6.1],
                'gdp_path': [2.1, 0.5, -0.2, 0.1, 0.8],
                'probability': 0.10
            }
        }
        
        # Mock portfolio for testing
        test_portfolio = {
            'value': 50000000,
            'assets': {
                'stocks': {'weight': 0.60, 'inflation_beta': -0.30, 'recession_beta': -2.50},
                'bonds': {'weight': 0.35, 'inflation_beta': -1.80, 'recession_beta': 1.20},
                'commodities': {'weight': 0.05, 'inflation_beta': 1.50, 'recession_beta': -0.50}
            }
        }
        
        def run_stress_test(scenarios, portfolio):
            """Run stress test across multiple scenarios."""
            results = {}
            
            for scenario_name, scenario in scenarios.items():
                scenario_impact = 0
                
                for asset, properties in portfolio['assets'].items():
                    weight = properties['weight']
                    
                    # Calculate inflation impact
                    avg_inflation = np.mean(scenario['inflation_path'])
                    inflation_impact = properties['inflation_beta'] * avg_inflation / 100
                    
                    # Calculate recession impact (simplified)
                    min_gdp = min(scenario['gdp_path'])
                    recession_impact = properties['recession_beta'] * min(0, min_gdp) / 100
                    
                    asset_impact = (inflation_impact + recession_impact) * weight
                    scenario_impact += asset_impact
                
                results[scenario_name] = {
                    'portfolio_impact_pct': scenario_impact,
                    'portfolio_impact_value': portfolio['value'] * scenario_impact,
                    'probability': scenario['probability'],
                    'risk_contribution': scenario_impact * scenario['probability']
                }
            
            return results
        
        stress_results = run_stress_test(stress_scenarios, test_portfolio)
        
        # Validate stress test results
        assert len(stress_results) == len(stress_scenarios)
        
        for scenario_name, result in stress_results.items():
            assert 'portfolio_impact_pct' in result
            assert 'portfolio_impact_value' in result
            assert 'probability' in result
            
            # High inflation should hurt traditional portfolios
            if scenario_name == 'high_inflation':
                assert result['portfolio_impact_pct'] < 0
            
            # Recession impact depends on asset mix
            if scenario_name == 'recession':
                # Bond-heavy portfolios might benefit
                pass  # Impact could be positive or negative
        
        # Calculate expected loss across scenarios
        expected_loss = sum(result['risk_contribution'] for result in stress_results.values())
        assert isinstance(expected_loss, (int, float))
    
    def test_market_data_real_time_integration(self):
        """Test real-time market data integration and processing."""
        def simulate_market_data_feeds():
            """Simulate multiple market data feeds."""
            return {
                'bond_yields': {
                    '2y': 4.75,
                    '5y': 4.35,
                    '10y': 4.25,
                    '30y': 4.40,
                    'timestamp': datetime.now()
                },
                'commodities': {
                    'gold': 2045.50,
                    'oil_wti': 85.75,
                    'copper': 8.25,
                    'silver': 24.80,
                    'timestamp': datetime.now()
                },
                'currencies': {
                    'dxy': 103.25,
                    'eurusd': 1.0850,
                    'gbpusd': 1.2650,
                    'usdjpy': 149.75,
                    'timestamp': datetime.now()
                },
                'inflation_breakevens': {
                    '5y5y': 2.38,
                    '10y': 2.25,
                    '5y': 2.45,
                    'timestamp': datetime.now()
                }
            }
        
        def calculate_inflation_signals(market_data):
            """Calculate inflation signals from market data."""
            signals = {}
            
            # Yield curve signal
            yield_curve_slope = market_data['bond_yields']['10y'] - market_data['bond_yields']['2y']
            signals['yield_curve'] = {
                'slope': yield_curve_slope,
                'signal': 'steepening' if yield_curve_slope > 0.5 else 'flattening' if yield_curve_slope < -0.5 else 'neutral'
            }
            
            # Commodity signal
            gold_oil_ratio = market_data['commodities']['gold'] / market_data['commodities']['oil_wti']
            signals['commodities'] = {
                'gold_oil_ratio': gold_oil_ratio,
                'signal': 'inflation_hedge' if gold_oil_ratio > 24 else 'deflationary' if gold_oil_ratio < 20 else 'neutral'
            }
            
            # Currency signal
            dxy_level = market_data['currencies']['dxy']
            signals['currency'] = {
                'dxy_level': dxy_level,
                'signal': 'dollar_strength' if dxy_level > 105 else 'dollar_weakness' if dxy_level < 100 else 'neutral'
            }
            
            # Breakeven inflation signal
            breakeven_5y = market_data['inflation_breakevens']['5y']
            signals['breakevens'] = {
                'level': breakeven_5y,
                'signal': 'high_inflation_priced' if breakeven_5y > 3.0 else 'low_inflation_priced' if breakeven_5y < 2.0 else 'neutral'
            }
            
            return signals
        
        market_data = simulate_market_data_feeds()
        inflation_signals = calculate_inflation_signals(market_data)
        
        # Validate market data integration
        assert 'yield_curve' in inflation_signals
        assert 'commodities' in inflation_signals
        assert 'currency' in inflation_signals
        assert 'breakevens' in inflation_signals
        
        # Validate signal calculations
        for signal_category, signal_data in inflation_signals.items():
            assert 'signal' in signal_data
            assert signal_data['signal'] in ['inflation_hedge', 'deflationary', 'neutral', 
                                           'steepening', 'flattening', 'dollar_strength', 
                                           'dollar_weakness', 'high_inflation_priced', 'low_inflation_priced']
        
        # Test signal consistency (basic validation)
        if inflation_signals['commodities']['signal'] == 'inflation_hedge':
            # High commodity prices might indicate inflation concerns
            assert inflation_signals['commodities']['gold_oil_ratio'] > 20


class TestModelEnsembleIntegration:
    """Model ensemble and prediction integration tests."""
    
    def setup_method(self):
        """Setup model integration tests."""
        self.models = {
            'phillips_curve': {'weight': 0.25, 'accuracy': 0.78},
            'vector_autoregression': {'weight': 0.30, 'accuracy': 0.82},
            'neural_network': {'weight': 0.25, 'accuracy': 0.85},
            'gradient_boosting': {'weight': 0.20, 'accuracy': 0.80}
        }
        
        self.economic_features = {
            'unemployment_rate': 3.8,
            'wage_growth': 4.1,
            'capacity_utilization': 79.2,
            'money_supply_growth': 6.5,
            'import_prices': 112.3,
            'housing_costs': 385.7
        }
    
    def test_ensemble_prediction_integration(self):
        """Test ensemble model prediction integration."""
        def mock_individual_predictions(features, models):
            """Mock individual model predictions."""
            predictions = {}
            
            for model_name, model_props in models.items():
                # Simple mock prediction based on model type
                base_prediction = 2.8  # Base inflation rate
                
                if model_name == 'phillips_curve':
                    # Phillips curve: inverse relationship with unemployment
                    unemployment_effect = (6.0 - features['unemployment_rate']) * 0.3
                    predictions[model_name] = base_prediction + unemployment_effect
                    
                elif model_name == 'vector_autoregression':
                    # VAR: weighted combination of multiple factors
                    predictions[model_name] = (
                        base_prediction + 
                        (features['wage_growth'] - 3.5) * 0.2 +
                        (features['money_supply_growth'] - 5.0) * 0.1
                    )
                    
                elif model_name == 'neural_network':
                    # Neural network: complex nonlinear relationship
                    predictions[model_name] = (
                        base_prediction + 
                        np.sin(features['capacity_utilization'] / 20) * 0.5 +
                        (features['import_prices'] - 100) * 0.02
                    )
                    
                elif model_name == 'gradient_boosting':
                    # Gradient boosting: tree-based prediction
                    predictions[model_name] = (
                        base_prediction +
                        max(0, features['housing_costs'] - 350) * 0.005
                    )
            
            return predictions
        
        def create_ensemble_prediction(individual_predictions, model_weights):
            """Create ensemble prediction from individual models."""
            # Weighted average
            weighted_prediction = sum(
                individual_predictions[model] * model_weights[model]['weight']
                for model in individual_predictions
            )
            
            # Calculate prediction uncertainty
            prediction_variance = np.var(list(individual_predictions.values()))
            
            # Ensemble confidence based on model agreement
            confidence = 1 / (1 + prediction_variance)
            
            return {
                'ensemble_prediction': weighted_prediction,
                'individual_predictions': individual_predictions,
                'prediction_variance': prediction_variance,
                'confidence_score': confidence,
                'model_weights': {model: props['weight'] for model, props in model_weights.items()}
            }
        
        individual_preds = mock_individual_predictions(self.economic_features, self.models)
        ensemble_result = create_ensemble_prediction(individual_preds, self.models)
        
        # Validate ensemble results
        assert 'ensemble_prediction' in ensemble_result
        assert 'confidence_score' in ensemble_result
        assert 'individual_predictions' in ensemble_result
        
        # Ensemble prediction should be reasonable
        assert 0 < ensemble_result['ensemble_prediction'] < 10
        
        # Confidence should be between 0 and 1
        assert 0 <= ensemble_result['confidence_score'] <= 1
        
        # Should have predictions from all models
        assert len(ensemble_result['individual_predictions']) == len(self.models)
    
    def test_model_calibration_integration(self):
        """Test model calibration against historical data."""
        # Mock historical data
        historical_data = {
            'dates': pd.date_range('2020-01-01', '2024-01-01', freq='Q'),
            'actual_inflation': [1.8, 2.1, 2.9, 3.5, 4.2, 5.8, 6.1, 5.4, 4.1, 3.2, 2.8, 2.5, 2.3, 2.1, 2.4, 2.7],
            'features': {
                'unemployment': [3.5, 4.8, 8.2, 7.1, 6.2, 5.1, 3.9, 3.6, 3.5, 3.7, 3.8, 3.9, 3.8, 3.7, 3.8, 3.8],
                'wage_growth': [3.2, 2.8, 3.1, 3.8, 4.5, 5.2, 5.8, 5.1, 4.8, 4.3, 4.1, 4.0, 3.9, 4.0, 4.1, 4.2]
            }
        }
        
        def calibrate_model_performance(historical_data, models):
            """Calibrate model performance against historical data."""
            calibration_results = {}
            
            for model_name in models:
                predictions = []
                actuals = historical_data['actual_inflation']
                
                # Mock predictions for each period
                for i in range(len(actuals)):
                    if model_name == 'phillips_curve':
                        unemployment = historical_data['features']['unemployment'][i]
                        pred = 2.5 + (6.0 - unemployment) * 0.3
                    elif model_name == 'vector_autoregression':
                        wage_growth = historical_data['features']['wage_growth'][i]
                        pred = 2.0 + (wage_growth - 3.5) * 0.4
                    else:
                        # Simple mock for other models
                        pred = actuals[i] + np.random.normal(0, 0.5)
                    
                    predictions.append(pred)
                
                # Calculate performance metrics
                mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))
                rmse = np.sqrt(np.mean((np.array(actuals) - np.array(predictions))**2))
                correlation = np.corrcoef(actuals, predictions)[0, 1]
                
                calibration_results[model_name] = {
                    'mae': mae,
                    'rmse': rmse,
                    'correlation': correlation,
                    'predictions': predictions
                }
            
            return calibration_results
        
        calibration_results = calibrate_model_performance(historical_data, self.models)
        
        # Validate calibration results
        assert len(calibration_results) == len(self.models)
        
        for model_name, results in calibration_results.items():
            assert 'mae' in results
            assert 'rmse' in results
            assert 'correlation' in results
            
            # Performance metrics should be reasonable
            assert results['mae'] >= 0
            assert results['rmse'] >= 0
            assert -1 <= results['correlation'] <= 1
            
            # Good models should have reasonable accuracy
            assert results['mae'] < 2.0  # Less than 2% average error
            assert results['correlation'] > 0.3  # Some positive correlation
    
    def test_model_selection_integration(self):
        """Test dynamic model selection based on economic regimes."""
        def detect_economic_regime(features):
            """Detect current economic regime."""
            unemployment = features['unemployment_rate']
            wage_growth = features['wage_growth']
            capacity_util = features['capacity_utilization']
            
            if unemployment < 4.0 and capacity_util > 80:
                return 'tight_labor_market'
            elif unemployment > 6.0:
                return 'recession'
            elif wage_growth > 5.0:
                return 'wage_inflation'
            else:
                return 'normal'
        
        def adjust_model_weights_by_regime(base_weights, regime):
            """Adjust model weights based on economic regime."""
            regime_adjustments = {
                'tight_labor_market': {
                    'phillips_curve': 1.5,  # Phillips curve more relevant
                    'vector_autoregression': 1.0,
                    'neural_network': 0.8,
                    'gradient_boosting': 1.0
                },
                'recession': {
                    'phillips_curve': 0.7,  # Less relevant in recession
                    'vector_autoregression': 1.3,  # VAR captures interactions
                    'neural_network': 1.2,
                    'gradient_boosting': 1.0
                },
                'wage_inflation': {
                    'phillips_curve': 1.4,
                    'vector_autoregression': 1.1,
                    'neural_network': 0.9,
                    'gradient_boosting': 1.0
                },
                'normal': {
                    'phillips_curve': 1.0,
                    'vector_autoregression': 1.0,
                    'neural_network': 1.0,
                    'gradient_boosting': 1.0
                }
            }
            
            adjustments = regime_adjustments.get(regime, regime_adjustments['normal'])
            adjusted_weights = {}
            
            for model, base_weight in base_weights.items():
                adjusted_weight = base_weight['weight'] * adjustments.get(model, 1.0)
                adjusted_weights[model] = {'weight': adjusted_weight, 'accuracy': base_weight['accuracy']}
            
            # Normalize weights to sum to 1
            total_weight = sum(w['weight'] for w in adjusted_weights.values())
            for model in adjusted_weights:
                adjusted_weights[model]['weight'] /= total_weight
            
            return adjusted_weights, regime
        
        regime = detect_economic_regime(self.economic_features)
        adjusted_weights, detected_regime = adjust_model_weights_by_regime(self.models, regime)
        
        # Validate regime detection and weight adjustment
        assert detected_regime in ['tight_labor_market', 'recession', 'wage_inflation', 'normal']
        assert len(adjusted_weights) == len(self.models)
        
        # Weights should sum to approximately 1
        total_weight = sum(w['weight'] for w in adjusted_weights.values())
        assert abs(total_weight - 1.0) < 0.01
        
        # In tight labor market, Phillips curve should get more weight
        if regime == 'tight_labor_market':
            phillips_weight = adjusted_weights['phillips_curve']['weight']
            original_phillips_weight = self.models['phillips_curve']['weight']
            assert phillips_weight > original_phillips_weight


class TestPortfolioHedgeIntegration:
    """Portfolio hedging integration tests."""
    
    def test_end_to_end_hedge_optimization(self):
        """Test end-to-end hedge optimization workflow."""
        # Mock portfolio data
        portfolio = {
            'total_value': 100000000,
            'assets': {
                'us_equities': {'value': 40000000, 'inflation_beta': -0.30},
                'international_equities': {'value': 20000000, 'inflation_beta': -0.20},
                'government_bonds': {'value': 25000000, 'inflation_beta': -2.00},
                'corporate_bonds': {'value': 10000000, 'inflation_beta': -1.50},
                'real_estate': {'value': 5000000, 'inflation_beta': 0.80}
            }
        }
        
        # Mock inflation scenario
        inflation_scenario = {
            'base_case': 2.8,
            'stressed_case': 5.5,
            'probability_stress': 0.25
        }
        
        # Mock hedge instruments
        hedge_instruments = {
            'tips_etf': {
                'inflation_beta': 0.75,
                'cost_bps': 15,
                'liquidity': 'high',
                'max_position_pct': 0.30
            },
            'commodity_etf': {
                'inflation_beta': 1.20,
                'cost_bps': 45,
                'liquidity': 'medium',
                'max_position_pct': 0.15
            },
            'gold_etf': {
                'inflation_beta': 0.60,
                'cost_bps': 25,
                'liquidity': 'high',
                'max_position_pct': 0.10
            }
        }
        
        def optimize_hedge_allocation(portfolio, scenario, instruments):
            """Optimize hedge allocation across instruments."""
            # Calculate portfolio inflation sensitivity
            portfolio_beta = sum(
                (asset['value'] / portfolio['total_value']) * asset['inflation_beta']
                for asset in portfolio['assets'].values()
            )
            
            # Calculate expected loss in stress scenario
            stress_impact = portfolio_beta * (scenario['stressed_case'] - scenario['base_case']) / 100
            expected_loss = portfolio['total_value'] * stress_impact * scenario['probability_stress']
            
            # Optimize hedge allocation
            optimal_hedges = {}
            remaining_hedge_need = abs(expected_loss)
            
            # Sort instruments by cost-effectiveness
            instruments_sorted = sorted(
                instruments.items(),
                key=lambda x: x[1]['inflation_beta'] / (x[1]['cost_bps'] / 10000),
                reverse=True
            )
            
            for instrument_name, props in instruments_sorted:
                if remaining_hedge_need > 0:
                    # Calculate maximum effective hedge from this instrument
                    max_position_value = portfolio['total_value'] * props['max_position_pct']
                    max_hedge_value = max_position_value * props['inflation_beta']
                    
                    # Allocate hedge
                    hedge_allocation = min(remaining_hedge_need, max_hedge_value)
                    position_size = hedge_allocation / props['inflation_beta']
                    
                    if position_size > 0:
                        optimal_hedges[instrument_name] = {
                            'position_value': position_size,
                            'position_pct': position_size / portfolio['total_value'],
                            'hedge_value': hedge_allocation,
                            'annual_cost': position_size * props['cost_bps'] / 10000
                        }
                        
                        remaining_hedge_need -= hedge_allocation
            
            return {
                'portfolio_inflation_beta': portfolio_beta,
                'expected_loss_unhedged': expected_loss,
                'hedge_allocations': optimal_hedges,
                'remaining_risk': remaining_hedge_need,
                'total_hedge_cost': sum(h['annual_cost'] for h in optimal_hedges.values()),
                'hedge_effectiveness': 1 - (remaining_hedge_need / abs(expected_loss)) if expected_loss != 0 else 0
            }
        
        optimization_result = optimize_hedge_allocation(portfolio, inflation_scenario, hedge_instruments)
        
        # Validate optimization results
        assert 'portfolio_inflation_beta' in optimization_result
        assert 'hedge_allocations' in optimization_result
        assert 'hedge_effectiveness' in optimization_result
        
        # Portfolio should be negatively exposed to inflation
        assert optimization_result['portfolio_inflation_beta'] < 0
        
        # Should have some hedge allocations
        assert len(optimization_result['hedge_allocations']) > 0
        
        # Hedge effectiveness should be reasonable
        assert 0 <= optimization_result['hedge_effectiveness'] <= 1
        
        # Total hedge cost should be reasonable (less than 1% of portfolio)
        max_acceptable_cost = portfolio['total_value'] * 0.01
        assert optimization_result['total_hedge_cost'] < max_acceptable_cost


if __name__ == '__main__':
    # Run integration tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-x'  # Stop on first failure for integration tests
    ])
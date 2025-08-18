"""
Performance Testing Suite for Macroeconomic Inflation Hedge Analytics Platform
===============================================================================

Performance, scalability, and load testing for economic modeling, scenario analysis,
hedging calculations, and macroeconomic data processing.

Author: Emilio Cardenas
License: MIT
"""

import pytest
import time
import threading
import multiprocessing
import asyncio
import sys
import os
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock
import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
import requests
import memory_profiler

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from main import app
    from advanced_inflation_analytics import MacroeconomicInflationPlatform
    from analytics_engine import AnalyticsEngine
    from data_manager import DataManager
    from ml_models import MLModelManager
except ImportError as e:
    print(f"Warning: Using mocks for performance testing: {e}")
    app = Mock()


class TestEconomicModelPerformance:
    """Economic model performance tests."""
    
    def setup_method(self):
        """Setup performance test environment."""
        self.client = app.test_client() if hasattr(app, 'test_client') else Mock()
        self.base_url = 'http://localhost:8001'
        self.sample_request = {
            'indicators': ['cpi', 'pce', 'gdp'],
            'forecast_horizon': 12,
            'scenario': 'base_case'
        }
    
    def test_inflation_forecast_response_time(self):
        """Test inflation forecast API response time."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
        
        # Warmup request
        self.client.post('/api/v1/forecast', json=self.sample_request)
        
        # Measure response time
        start_time = time.perf_counter()
        response = self.client.post('/api/v1/forecast', json=self.sample_request)
        end_time = time.perf_counter()
        
        response_time = end_time - start_time
        
        # Performance requirements
        assert response_time < 2.0  # Should respond within 2 seconds
        
        if hasattr(response, 'status_code'):
            assert response.status_code in [200, 503]
    
    def test_concurrent_forecast_requests_performance(self):
        """Test concurrent forecast requests performance."""
        if not hasattr(self.client, 'post'):
            pytest.skip("Client not available")
        
        num_concurrent_requests = 10
        response_times = []
        successful_requests = 0
        
        def make_request():
            nonlocal successful_requests
            start_time = time.perf_counter()
            try:
                response = self.client.post('/api/v1/forecast', json=self.sample_request)
                end_time = time.perf_counter()
                response_times.append(end_time - start_time)
                if hasattr(response, 'status_code') and response.status_code == 200:
                    successful_requests += 1
            except Exception:
                pass
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=num_concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent_requests)]
            
            for future in futures:
                future.result()
        
        # Performance analysis
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # Performance requirements
            assert avg_response_time < 3.0  # Average under 3 seconds
            assert max_response_time < 8.0  # No request over 8 seconds
            assert successful_requests >= num_concurrent_requests * 0.7  # 70% success rate
    
    def test_large_dataset_processing_performance(self):
        """Test performance with large economic datasets."""
        # Generate large dataset
        dataset_sizes = [1000, 5000, 10000, 25000]
        
        for size in dataset_sizes:
            # Generate economic time series data
            start_time = time.perf_counter()
            
            economic_data = {
                'dates': pd.date_range('2000-01-01', periods=size, freq='D'),
                'cpi': np.random.normal(2.5, 0.8, size).cumsum() + 100,
                'pce': np.random.normal(2.2, 0.7, size).cumsum() + 95,
                'unemployment': np.random.normal(5.0, 1.5, size),
                'fed_funds': np.random.normal(3.5, 2.0, size),
                'gdp_growth': np.random.normal(2.0, 1.2, size)
            }
            
            df = pd.DataFrame(economic_data)
            
            # Mock data processing operations
            def process_economic_data(df):
                """Process economic data with typical operations."""
                # Calculate rolling averages
                df['cpi_ma'] = df['cpi'].rolling(window=30).mean()
                df['pce_ma'] = df['pce'].rolling(window=30).mean()
                
                # Calculate inflation rates
                df['inflation_rate'] = df['cpi'].pct_change(periods=12) * 100
                df['core_inflation'] = df['pce'].pct_change(periods=12) * 100
                
                # Calculate correlations
                correlation_matrix = df[['cpi', 'pce', 'unemployment', 'fed_funds', 'gdp_growth']].corr()
                
                # Simple forecasting
                recent_data = df.tail(100)
                forecast = {
                    'cpi_trend': recent_data['cpi'].mean(),
                    'inflation_forecast': recent_data['inflation_rate'].mean()
                }
                
                return {
                    'processed_data': df,
                    'correlations': correlation_matrix,
                    'forecast': forecast
                }
            
            result = process_economic_data(df)
            end_time = time.perf_counter()
            
            processing_time = end_time - start_time
            throughput = size / processing_time  # Records per second
            
            # Performance requirements
            assert processing_time < size * 0.001  # Less than 1ms per record
            assert throughput > 1000  # At least 1000 records per second
            
            # Validate results
            assert len(result['processed_data']) == size
            assert 'correlations' in result
            assert 'forecast' in result
    
    def test_scenario_analysis_performance(self):
        """Test scenario analysis performance with multiple scenarios."""
        # Define multiple economic scenarios
        scenarios = {
            'base_case': {'inflation': 2.8, 'gdp': 2.1, 'unemployment': 3.8},
            'high_inflation': {'inflation': 5.5, 'gdp': 1.2, 'unemployment': 4.8},
            'recession': {'inflation': 1.5, 'gdp': -1.8, 'unemployment': 7.2},
            'stagflation': {'inflation': 6.2, 'gdp': 0.3, 'unemployment': 6.5},
            'recovery': {'inflation': 3.1, 'gdp': 4.2, 'unemployment': 3.2}
        }
        
        # Mock portfolio for scenario analysis
        portfolio = {
            'assets': {
                'stocks': {'weight': 0.60, 'inflation_beta': -0.25, 'gdp_beta': 1.8},
                'bonds': {'weight': 0.30, 'inflation_beta': -1.50, 'gdp_beta': -0.2},
                'commodities': {'weight': 0.05, 'inflation_beta': 1.25, 'gdp_beta': 0.8},
                'real_estate': {'weight': 0.05, 'inflation_beta': 0.75, 'gdp_beta': 1.2}
            },
            'total_value': 100000000
        }
        
        def run_scenario_analysis(scenarios, portfolio, num_iterations=1000):
            """Run scenario analysis with Monte Carlo simulation."""
            results = {}
            
            for scenario_name, scenario_params in scenarios.items():
                scenario_results = []
                
                for _ in range(num_iterations):
                    # Add noise to scenario parameters
                    noisy_inflation = scenario_params['inflation'] + np.random.normal(0, 0.3)
                    noisy_gdp = scenario_params['gdp'] + np.random.normal(0, 0.5)
                    
                    # Calculate portfolio impact
                    portfolio_impact = 0
                    for asset, props in portfolio['assets'].items():
                        inflation_impact = props['inflation_beta'] * noisy_inflation / 100
                        gdp_impact = props['gdp_beta'] * noisy_gdp / 100
                        
                        total_asset_impact = (inflation_impact + gdp_impact) * props['weight']
                        portfolio_impact += total_asset_impact
                    
                    scenario_results.append(portfolio_impact)
                
                # Calculate statistics
                results[scenario_name] = {
                    'mean_impact': np.mean(scenario_results),
                    'std_impact': np.std(scenario_results),
                    'var_95': np.percentile(scenario_results, 5),
                    'var_99': np.percentile(scenario_results, 1)
                }
            
            return results
        
        # Measure performance
        start_time = time.perf_counter()
        analysis_results = run_scenario_analysis(scenarios, portfolio, num_iterations=1000)
        end_time = time.perf_counter()
        
        analysis_time = end_time - start_time
        scenarios_per_second = len(scenarios) / analysis_time
        
        # Performance requirements
        assert analysis_time < 5.0  # Complete within 5 seconds
        assert scenarios_per_second > 0.5  # At least 0.5 scenarios per second
        
        # Validate results
        assert len(analysis_results) == len(scenarios)
        for scenario_result in analysis_results.values():
            assert 'mean_impact' in scenario_result
            assert 'var_95' in scenario_result
    
    def test_hedge_optimization_performance(self):
        """Test hedge optimization algorithm performance."""
        # Mock hedge instruments with various complexities
        hedge_instruments = {
            'tips_5y': {'beta': 0.85, 'cost': 0.0025, 'liquidity': 1.0},
            'tips_10y': {'beta': 0.78, 'cost': 0.0022, 'liquidity': 0.95},
            'commodities': {'beta': 1.25, 'cost': 0.0080, 'liquidity': 0.75},
            'gold': {'beta': 0.60, 'cost': 0.0050, 'liquidity': 0.90},
            'real_estate': {'beta': 0.70, 'cost': 0.0150, 'liquidity': 0.30},
            'inflation_swaps_2y': {'beta': 1.00, 'cost': 0.0035, 'liquidity': 0.60},
            'inflation_swaps_5y': {'beta': 0.95, 'cost': 0.0030, 'liquidity': 0.65},
            'energy_etf': {'beta': 1.15, 'cost': 0.0065, 'liquidity': 0.80}
        }
        
        # Portfolio exposure to optimize
        portfolio_exposure = {
            'inflation_beta': -0.45,
            'target_beta': -0.10,  # Reduce exposure
            'portfolio_value': 500000000,
            'max_hedge_allocation': 0.25  # Max 25% in hedges
        }
        
        def optimize_hedge_portfolio(instruments, exposure, optimization_method='quadratic'):
            """Optimize hedge portfolio using various methods."""
            if optimization_method == 'quadratic':
                # Simple quadratic optimization
                target_hedge = abs(exposure['inflation_beta'] - exposure['target_beta'])
                max_hedge_value = exposure['portfolio_value'] * exposure['max_hedge_allocation']
                
                # Greedy allocation by efficiency
                instruments_sorted = sorted(
                    instruments.items(),
                    key=lambda x: x[1]['beta'] / (x[1]['cost'] * (2 - x[1]['liquidity'])),
                    reverse=True
                )
                
                allocation = {}
                remaining_hedge = target_hedge
                
                for instrument, props in instruments_sorted:
                    if remaining_hedge > 0:
                        # Calculate optimal allocation
                        max_instrument_allocation = min(
                            max_hedge_value * 0.4,  # Max 40% in single instrument
                            remaining_hedge / props['beta'] * exposure['portfolio_value']
                        )
                        
                        if max_instrument_allocation > 0:
                            allocation[instrument] = max_instrument_allocation / exposure['portfolio_value']
                            remaining_hedge -= (allocation[instrument] * props['beta'])
                
                return allocation
            
            elif optimization_method == 'monte_carlo':
                # Monte Carlo optimization
                best_allocation = None
                best_score = float('inf')
                
                for _ in range(1000):  # 1000 random trials
                    allocation = {}
                    total_allocation = 0
                    
                    for instrument in instruments:
                        if total_allocation < exposure['max_hedge_allocation']:
                            random_allocation = np.random.uniform(0, 0.1)  # Up to 10% each
                            if total_allocation + random_allocation <= exposure['max_hedge_allocation']:
                                allocation[instrument] = random_allocation
                                total_allocation += random_allocation
                    
                    # Calculate portfolio beta after hedging
                    hedge_beta = sum(
                        allocation.get(inst, 0) * props['beta']
                        for inst, props in instruments.items()
                    )
                    
                    final_beta = exposure['inflation_beta'] + hedge_beta
                    
                    # Calculate cost
                    total_cost = sum(
                        allocation.get(inst, 0) * props['cost']
                        for inst, props in instruments.items()
                    )
                    
                    # Objective: minimize distance from target + cost penalty
                    score = abs(final_beta - exposure['target_beta']) + total_cost * 10
                    
                    if score < best_score:
                        best_score = score
                        best_allocation = allocation
                
                return best_allocation or {}
            
            else:
                return {}
        
        # Test different optimization methods
        methods = ['quadratic', 'monte_carlo']
        performance_results = {}
        
        for method in methods:
            start_time = time.perf_counter()
            allocation = optimize_hedge_portfolio(hedge_instruments, portfolio_exposure, method)
            end_time = time.perf_counter()
            
            optimization_time = end_time - start_time
            performance_results[method] = {
                'time': optimization_time,
                'allocation': allocation,
                'instruments_used': len(allocation)
            }
        
        # Performance requirements
        for method, result in performance_results.items():
            if method == 'quadratic':
                assert result['time'] < 0.1  # Very fast
            elif method == 'monte_carlo':
                assert result['time'] < 2.0  # Reasonable for Monte Carlo
            
            assert len(result['allocation']) > 0  # Should find some allocation


class TestDataProcessingPerformance:
    """Data processing and analytics performance tests."""
    
    def test_economic_data_ingestion_performance(self):
        """Test performance of economic data ingestion."""
        # Simulate various data sizes
        data_sizes = [1000, 5000, 10000, 50000]
        
        for size in data_sizes:
            # Generate mock economic data
            start_time = time.perf_counter()
            
            economic_indicators = {
                'cpi_data': np.random.normal(287.5, 15.0, size),
                'pce_data': np.random.normal(275.2, 12.0, size),
                'unemployment': np.random.normal(3.8, 1.2, size),
                'gdp_growth': np.random.normal(2.1, 1.8, size),
                'fed_funds': np.random.normal(5.25, 1.5, size),
                'timestamps': pd.date_range('2000-01-01', periods=size, freq='D')
            }
            
            # Data processing pipeline
            def process_economic_indicators(data):
                """Process economic indicators through typical pipeline."""
                df = pd.DataFrame(data)
                
                # Data cleaning
                df = df.dropna()
                
                # Calculate derived indicators
                df['inflation_rate'] = df['cpi_data'].pct_change(periods=252) * 100  # Annual
                df['core_inflation'] = df['pce_data'].pct_change(periods=252) * 100
                df['real_gdp'] = df['gdp_growth'] - df['inflation_rate'] / 100
                
                # Statistical aggregations
                rolling_stats = {
                    'cpi_ma_30': df['cpi_data'].rolling(30).mean(),
                    'cpi_std_30': df['cpi_data'].rolling(30).std(),
                    'inflation_ma_90': df['inflation_rate'].rolling(90).mean()
                }
                
                # Correlation analysis
                correlation_matrix = df[['cpi_data', 'pce_data', 'unemployment', 'gdp_growth']].corr()
                
                return {
                    'processed_data': df,
                    'rolling_stats': rolling_stats,
                    'correlations': correlation_matrix.to_dict(),
                    'records_processed': len(df)
                }
            
            result = process_economic_indicators(economic_indicators)
            
            end_time = time.perf_counter()
            processing_time = end_time - start_time
            throughput = size / processing_time
            
            # Performance requirements
            assert processing_time < size * 0.0005  # Less than 0.5ms per record
            assert throughput > 2000  # At least 2000 records per second
            
            # Validate processing results
            assert result['records_processed'] > 0
            assert 'correlations' in result
    
    def test_time_series_analysis_performance(self):
        """Test time series analysis performance."""
        # Generate time series data
        time_periods = [100, 500, 1000, 2500]
        
        for periods in time_periods:
            # Create synthetic economic time series
            dates = pd.date_range('2020-01-01', periods=periods, freq='D')
            
            # Generate correlated economic series
            np.random.seed(42)
            base_trend = np.cumsum(np.random.normal(0, 0.1, periods))
            
            economic_series = {
                'inflation': base_trend + np.random.normal(2.5, 0.5, periods),
                'gdp_growth': -0.5 * base_trend + np.random.normal(2.0, 1.0, periods),
                'unemployment': 0.3 * base_trend + np.random.normal(4.0, 0.8, periods),
                'interest_rates': 0.8 * base_trend + np.random.normal(3.5, 1.2, periods)
            }
            
            df = pd.DataFrame(economic_series, index=dates)
            
            def comprehensive_time_series_analysis(ts_data):
                """Perform comprehensive time series analysis."""
                results = {}
                
                # Trend analysis
                for col in ts_data.columns:
                    series = ts_data[col]
                    
                    # Basic statistics
                    results[f'{col}_mean'] = series.mean()
                    results[f'{col}_std'] = series.std()
                    results[f'{col}_trend'] = np.polyfit(range(len(series)), series, 1)[0]
                    
                    # Autocorrelation
                    results[f'{col}_autocorr_1'] = series.autocorr(lag=1)
                    results[f'{col}_autocorr_30'] = series.autocorr(lag=30)
                    
                    # Rolling statistics
                    results[f'{col}_rolling_mean'] = series.rolling(30).mean().iloc[-1]
                    results[f'{col}_rolling_vol'] = series.rolling(30).std().iloc[-1]
                
                # Cross-correlations
                correlation_matrix = ts_data.corr()
                results['correlation_matrix'] = correlation_matrix.to_dict()
                
                # Cointegration tests (simplified)
                results['cointegration_pairs'] = []
                for i, col1 in enumerate(ts_data.columns):
                    for col2 in ts_data.columns[i+1:]:
                        corr = correlation_matrix.loc[col1, col2]
                        if abs(corr) > 0.7:
                            results['cointegration_pairs'].append((col1, col2, corr))
                
                return results
            
            start_time = time.perf_counter()
            analysis_results = comprehensive_time_series_analysis(df)
            end_time = time.perf_counter()
            
            analysis_time = end_time - start_time
            
            # Performance requirements
            assert analysis_time < periods * 0.001  # Less than 1ms per period
            
            # Validate analysis results
            assert len(analysis_results) > 0
            assert 'correlation_matrix' in analysis_results
            assert 'cointegration_pairs' in analysis_results
    
    def test_monte_carlo_simulation_performance(self):
        """Test Monte Carlo simulation performance."""
        simulation_sizes = [1000, 5000, 10000, 25000]
        
        for sim_size in simulation_sizes:
            # Economic parameters for simulation
            parameters = {
                'inflation_mean': 2.8,
                'inflation_vol': 1.2,
                'gdp_mean': 2.1,
                'gdp_vol': 1.8,
                'correlation': -0.3,
                'time_horizon': 12  # months
            }
            
            def run_economic_monte_carlo(params, n_simulations):
                """Run Monte Carlo simulation for economic scenarios."""
                results = []
                
                # Correlation matrix
                corr_matrix = np.array([[1.0, params['correlation']], 
                                      [params['correlation'], 1.0]])
                
                for _ in range(n_simulations):
                    # Generate correlated random variables
                    random_vars = np.random.multivariate_normal([0, 0], corr_matrix)
                    
                    # Generate paths
                    inflation_path = []
                    gdp_path = []
                    
                    current_inflation = params['inflation_mean']
                    current_gdp = params['gdp_mean']
                    
                    for month in range(params['time_horizon']):
                        # Random shocks
                        inflation_shock = random_vars[0] * params['inflation_vol'] / np.sqrt(12)
                        gdp_shock = random_vars[1] * params['gdp_vol'] / np.sqrt(12)
                        
                        # Update values
                        current_inflation += inflation_shock
                        current_gdp += gdp_shock
                        
                        inflation_path.append(current_inflation)
                        gdp_path.append(current_gdp)
                    
                    # Calculate simulation metrics
                    avg_inflation = np.mean(inflation_path)
                    avg_gdp = np.mean(gdp_path)
                    inflation_volatility = np.std(inflation_path)
                    
                    results.append({
                        'avg_inflation': avg_inflation,
                        'avg_gdp': avg_gdp,
                        'inflation_vol': inflation_volatility,
                        'final_inflation': inflation_path[-1],
                        'final_gdp': gdp_path[-1]
                    })
                
                return results
            
            start_time = time.perf_counter()
            mc_results = run_economic_monte_carlo(parameters, sim_size)
            end_time = time.perf_counter()
            
            simulation_time = end_time - start_time
            simulations_per_second = sim_size / simulation_time
            
            # Performance requirements
            assert simulation_time < sim_size * 0.001  # Less than 1ms per simulation
            assert simulations_per_second > 1000  # At least 1000 sims/sec
            
            # Validate simulation results
            assert len(mc_results) == sim_size
            assert all('avg_inflation' in result for result in mc_results[:5])


class TestMemoryAndResourceManagement:
    """Memory and resource management performance tests."""
    
    def test_memory_usage_optimization(self):
        """Test memory usage during large data processing."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Process large economic datasets
        large_dataset_size = 100000
        
        def memory_efficient_processing(size):
            """Process data in memory-efficient chunks."""
            chunk_size = 10000
            results = []
            
            for start_idx in range(0, size, chunk_size):
                end_idx = min(start_idx + chunk_size, size)
                current_size = end_idx - start_idx
                
                # Generate chunk of data
                chunk_data = {
                    'cpi': np.random.normal(287.5, 15.0, current_size),
                    'pce': np.random.normal(275.2, 12.0, current_size),
                    'unemployment': np.random.normal(3.8, 1.2, current_size)
                }
                
                # Process chunk
                chunk_df = pd.DataFrame(chunk_data)
                chunk_result = {
                    'mean_cpi': chunk_df['cpi'].mean(),
                    'mean_pce': chunk_df['pce'].mean(),
                    'correlation': chunk_df['cpi'].corr(chunk_df['pce'])
                }
                
                results.append(chunk_result)
                
                # Clear chunk data
                del chunk_data, chunk_df
                gc.collect()
            
            return results
        
        # Monitor memory during processing
        gc.collect()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        results = memory_efficient_processing(large_dataset_size)
        
        gc.collect()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = memory_after - memory_before
        
        # Memory requirements
        assert len(results) > 0
        assert memory_increase < 200  # Should not use more than 200MB additional
    
    def test_concurrent_processing_scalability(self):
        """Test scalability with concurrent processing."""
        def cpu_intensive_calculation(data_size):
            """CPU intensive economic calculation."""
            # Generate economic data
            inflation_data = np.random.normal(2.8, 1.2, data_size)
            gdp_data = np.random.normal(2.1, 1.8, data_size)
            
            # Perform calculations
            correlation = np.corrcoef(inflation_data, gdp_data)[0, 1]
            
            # Monte Carlo simulation
            simulations = []
            for _ in range(100):
                sim_inflation = np.random.choice(inflation_data, 12)  # 1 year
                sim_gdp = np.random.choice(gdp_data, 12)
                
                avg_inflation = np.mean(sim_inflation)
                avg_gdp = np.mean(sim_gdp)
                
                simulations.append({'inflation': avg_inflation, 'gdp': avg_gdp})
            
            return {
                'correlation': correlation,
                'simulation_results': simulations,
                'processed_size': data_size
            }
        
        data_size = 10000
        
        # Test sequential processing
        start_time = time.perf_counter()
        sequential_results = [cpu_intensive_calculation(data_size) for _ in range(4)]
        sequential_time = time.perf_counter() - start_time
        
        # Test parallel processing
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_intensive_calculation, data_size) for _ in range(4)]
            parallel_results = [future.result() for future in futures]
        parallel_time = time.perf_counter() - start_time
        
        # Validate results
        assert len(sequential_results) == 4
        assert len(parallel_results) == 4
        
        # Performance comparison
        speedup = sequential_time / parallel_time
        
        # Should see some speedup with parallel processing
        assert speedup > 1.2  # At least 20% improvement
        assert parallel_time < sequential_time


@pytest.mark.slow
class TestStressAndLoadTesting:
    """Stress testing and load testing for economic analytics."""
    
    def test_sustained_economic_analysis_load(self):
        """Test sustained load for economic analysis."""
        if not hasattr(app, 'test_client'):
            pytest.skip("App test client not available")
        
        client = app.test_client()
        duration = 30  # seconds
        request_interval = 1.0  # seconds between requests
        
        start_time = time.time()
        request_count = 0
        error_count = 0
        response_times = []
        
        sample_request = {
            'indicators': ['cpi', 'gdp'],
            'forecast_horizon': 6
        }
        
        while time.time() - start_time < duration:
            try:
                request_start = time.perf_counter()
                response = client.post('/api/v1/economic_analysis', json=sample_request)
                request_end = time.perf_counter()
                
                response_times.append(request_end - request_start)
                request_count += 1
                
                if hasattr(response, 'status_code') and response.status_code != 200:
                    error_count += 1
                    
            except Exception:
                error_count += 1
            
            time.sleep(request_interval)
        
        # Performance analysis
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            error_rate = error_count / request_count if request_count > 0 else 1
            
            # Performance requirements for sustained load
            assert avg_response_time < 3.0  # Average response time under 3 seconds
            assert max_response_time < 10.0  # No response over 10 seconds
            assert error_rate < 0.2  # Less than 20% error rate
    
    def test_memory_leak_detection_economic_models(self):
        """Test for memory leaks during extended model operations."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Simulate extended economic model operations
        for i in range(500):
            # Mock economic modeling operations
            economic_data = {
                'inflation_series': np.random.normal(2.8, 1.2, 1000),
                'gdp_series': np.random.normal(2.1, 1.8, 1000),
                'unemployment_series': np.random.normal(3.8, 1.5, 1000)
            }
            
            # Process economic data
            df = pd.DataFrame(economic_data)
            correlations = df.corr()
            forecast = df.tail(50).mean()
            
            # Simulate model training/prediction
            X = df[['inflation_series', 'gdp_series']].values
            y = df['unemployment_series'].values
            
            # Simple linear regression equivalent
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            predictions = X @ beta
            
            # Clean up
            del economic_data, df, correlations, forecast, X, y, beta, predictions
            
            # Check memory every 100 iterations
            if i % 100 == 0:
                gc.collect()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory should not increase significantly
                assert memory_increase < 150  # Less than 150MB increase
        
        # Final memory check
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        total_memory_increase = final_memory - initial_memory
        
        # Should not have significant memory leak
        assert total_memory_increase < 100  # Less than 100MB total increase


if __name__ == '__main__':
    # Run performance tests
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-m', 'not slow'  # Skip slow tests by default
    ])
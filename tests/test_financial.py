"""
Data Analysis Tool - Tests for Financial Module

This module contains tests for the FinancialAnalyzer class.
"""

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import the module to test
from dataanalysistool.financial import FinancialAnalyzer

class TestFinancialAnalyzer(unittest.TestCase):
    """Test cases for FinancialAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = FinancialAnalyzer()
        
        # Create test stock data
        index = pd.date_range(start='2020-01-01', periods=100, freq='D')
        self.test_df = pd.DataFrame({
            'Open': np.linspace(100, 150, 100) + np.random.normal(0, 5, 100),
            'High': np.linspace(105, 155, 100) + np.random.normal(0, 5, 100),
            'Low': np.linspace(95, 145, 100) + np.random.normal(0, 5, 100),
            'Close': np.linspace(100, 150, 100) + np.random.normal(0, 5, 100),
            'Volume': np.random.randint(1000, 10000, 100),
            'Adj Close': np.linspace(100, 150, 100) + np.random.normal(0, 5, 100)
        }, index=index)
        
        # Ensure High is the highest and Low is the lowest
        for i in range(len(self.test_df)):
            high = max(self.test_df.iloc[i]['Open'], self.test_df.iloc[i]['Close'], self.test_df.iloc[i]['High'])
            low = min(self.test_df.iloc[i]['Open'], self.test_df.iloc[i]['Close'], self.test_df.iloc[i]['Low'])
            self.test_df.iloc[i, self.test_df.columns.get_loc('High')] = high
            self.test_df.iloc[i, self.test_df.columns.get_loc('Low')] = low
        
        # Set data for analyzer
        self.analyzer.set_data(self.test_df, ticker='TEST')
    
    def tearDown(self):
        """Tear down test fixtures."""
        plt.close('all')  # Close all matplotlib figures
    
    def test_set_data(self):
        """Test setting data."""
        # Create new data
        new_df = pd.DataFrame({
            'Close': [100, 101, 102],
            'Volume': [1000, 1100, 1200]
        }, index=pd.date_range(start='2020-01-01', periods=3))
        
        # Set new data
        self.analyzer.set_data(new_df, ticker='NEW')
        
        # Check that data was set correctly
        self.assertEqual(self.analyzer.data.shape, (3, 2))
        self.assertEqual(self.analyzer.ticker, 'NEW')
        self.assertEqual(self.analyzer.results, {})
    
    def test_calculate_returns(self):
        """Test calculating returns."""
        # Calculate returns
        returns = self.analyzer.calculate_returns(price_column='Close', period='daily')
        
        # Check that returns were calculated correctly
        self.assertIsInstance(returns, pd.Series)
        self.assertEqual(len(returns), 100)
        self.assertTrue('daily_returns' in self.analyzer.results)
    
    def test_calculate_cumulative_returns(self):
        """Test calculating cumulative returns."""
        # Calculate cumulative returns
        cum_return = self.analyzer.calculate_cumulative_returns(price_column='Close')
        
        # Check that cumulative return was calculated correctly
        self.assertIsInstance(cum_return, float)
        self.assertTrue('cumulative_return' in self.analyzer.results)
    
    def test_calculate_volatility(self):
        """Test calculating volatility."""
        # Calculate returns first
        self.analyzer.calculate_returns(price_column='Close', period='daily')
        
        # Calculate volatility
        volatility = self.analyzer.calculate_volatility(annualize=True)
        
        # Check that volatility was calculated correctly
        self.assertIsInstance(volatility, float)
        self.assertTrue('volatility' in self.analyzer.results)
    
    def test_calculate_sharpe_ratio(self):
        """Test calculating Sharpe ratio."""
        # Calculate returns first
        self.analyzer.calculate_returns(price_column='Close', period='daily')
        
        # Calculate Sharpe ratio
        sharpe = self.analyzer.calculate_sharpe_ratio(risk_free_rate=0.02)
        
        # Check that Sharpe ratio was calculated correctly
        self.assertIsInstance(sharpe, float)
        self.assertTrue('sharpe_ratio' in self.analyzer.results)
    
    def test_calculate_drawdown(self):
        """Test calculating drawdown."""
        # Calculate drawdown
        drawdown, max_drawdown, max_drawdown_duration = self.analyzer.calculate_drawdown(price_column='Close')
        
        # Check that drawdown was calculated correctly
        self.assertIsInstance(drawdown, pd.Series)
        self.assertIsInstance(max_drawdown, float)
        self.assertIsInstance(max_drawdown_duration, pd.Timedelta)
        self.assertTrue('drawdown' in self.analyzer.results)
        self.assertTrue('max_drawdown' in self.analyzer.results)
        self.assertTrue('max_drawdown_duration' in self.analyzer.results)
    
    def test_calculate_beta(self):
        """Test calculating beta."""
        # Create market data
        market_data = pd.DataFrame({
            'Close': np.linspace(1000, 1200, 100) + np.random.normal(0, 20, 100)
        }, index=self.test_df.index)
        
        # Calculate beta
        beta = self.analyzer.calculate_beta(market_data, price_column='Close', market_column='Close')
        
        # Check that beta was calculated correctly
        self.assertIsInstance(beta, float)
        self.assertTrue('beta' in self.analyzer.results)
    
    def test_calculate_alpha(self):
        """Test calculating alpha."""
        # Create market data
        market_data = pd.DataFrame({
            'Close': np.linspace(1000, 1200, 100) + np.random.normal(0, 20, 100)
        }, index=self.test_df.index)
        
        # Calculate beta first
        self.analyzer.calculate_beta(market_data, price_column='Close', market_column='Close')
        
        # Calculate alpha
        alpha = self.analyzer.calculate_alpha(market_data, risk_free_rate=0.02, price_column='Close', market_column='Close')
        
        # Check that alpha was calculated correctly
        self.assertIsInstance(alpha, float)
        self.assertTrue('alpha' in self.analyzer.results)
    
    def test_calculate_moving_averages(self):
        """Test calculating moving averages."""
        # Calculate moving averages
        ma_df = self.analyzer.calculate_moving_averages(price_column='Close', windows=[20, 50])
        
        # Check that moving averages were calculated correctly
        self.assertIsInstance(ma_df, pd.DataFrame)
        self.assertEqual(ma_df.shape, (100, 3))  # Close, MA_20, MA_50
        self.assertTrue('Close' in ma_df.columns)
        self.assertTrue('MA_20' in ma_df.columns)
        self.assertTrue('MA_50' in ma_df.columns)
        self.assertTrue('moving_averages' in self.analyzer.results)
    
    def test_calculate_rsi(self):
        """Test calculating RSI."""
        # Calculate RSI
        rsi = self.analyzer.calculate_rsi(price_column='Close', window=14)
        
        # Check that RSI was calculated correctly
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), 100)
        self.assertTrue('rsi' in self.analyzer.results)
    
    def test_calculate_macd(self):
        """Test calculating MACD."""
        # Calculate MACD
        macd_line, signal_line, histogram = self.analyzer.calculate_macd(price_column='Close')
        
        # Check that MACD was calculated correctly
        self.assertIsInstance(macd_line, pd.Series)
        self.assertIsInstance(signal_line, pd.Series)
        self.assertIsInstance(histogram, pd.Series)
        self.assertEqual(len(macd_line), 100)
        self.assertEqual(len(signal_line), 100)
        self.assertEqual(len(histogram), 100)
        self.assertTrue('macd' in self.analyzer.results)
    
    def test_calculate_bollinger_bands(self):
        """Test calculating Bollinger Bands."""
        # Calculate Bollinger Bands
        middle_band, upper_band, lower_band = self.analyzer.calculate_bollinger_bands(price_column='Close')
        
        # Check that Bollinger Bands were calculated correctly
        self.assertIsInstance(middle_band, pd.Series)
        self.assertIsInstance(upper_band, pd.Series)
        self.assertIsInstance(lower_band, pd.Series)
        self.assertEqual(len(middle_band), 100)
        self.assertEqual(len(upper_band), 100)
        self.assertEqual(len(lower_band), 100)
        self.assertTrue('bollinger_bands' in self.analyzer.results)
    
    def test_calculate_performance_metrics(self):
        """Test calculating performance metrics."""
        # Create market data
        market_data = pd.DataFrame({
            'Close': np.linspace(1000, 1200, 100) + np.random.normal(0, 20, 100)
        }, index=self.test_df.index)
        
        # Calculate performance metrics
        metrics = self.analyzer.calculate_performance_metrics(benchmark_data=market_data, risk_free_rate=0.02)
        
        # Check that performance metrics were calculated correctly
        self.assertIsInstance(metrics, dict)
        self.assertTrue('ticker' in metrics)
        self.assertTrue('cumulative_return' in metrics)
        self.assertTrue('annualized_return' in metrics)
        self.assertTrue('annualized_volatility' in metrics)
        self.assertTrue('sharpe_ratio' in metrics)
        self.assertTrue('max_drawdown' in metrics)
        self.assertTrue('beta' in metrics)
        self.assertTrue('alpha' in metrics)
        self.assertTrue('performance_metrics' in self.analyzer.results)
    
    def test_plot_price_history_static(self):
        """Test plotting price history (static)."""
        # Create plot
        fig = self.analyzer.plot_price_history(price_column='Close', interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_price_history_interactive(self):
        """Test plotting price history (interactive)."""
        # Create plot
        fig = self.analyzer.plot_price_history(price_column='Close', interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_plot_returns_distribution_static(self):
        """Test plotting returns distribution (static)."""
        # Calculate returns first
        self.analyzer.calculate_returns(price_column='Close', period='daily')
        
        # Create plot
        fig = self.analyzer.plot_returns_distribution(interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_returns_distribution_interactive(self):
        """Test plotting returns distribution (interactive)."""
        # Calculate returns first
        self.analyzer.calculate_returns(price_column='Close', period='daily')
        
        # Create plot
        fig = self.analyzer.plot_returns_distribution(interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_plot_cumulative_returns_static(self):
        """Test plotting cumulative returns (static)."""
        # Calculate returns first
        self.analyzer.calculate_returns(price_column='Close', period='daily')
        
        # Create plot
        fig = self.analyzer.plot_cumulative_returns(interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_cumulative_returns_interactive(self):
        """Test plotting cumulative returns (interactive)."""
        # Calculate returns first
        self.analyzer.calculate_returns(price_column='Close', period='daily')
        
        # Create plot
        fig = self.analyzer.plot_cumulative_returns(interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_plot_drawdown_static(self):
        """Test plotting drawdown (static)."""
        # Calculate drawdown first
        self.analyzer.calculate_drawdown(price_column='Close')
        
        # Create plot
        fig = self.analyzer.plot_drawdown(interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_drawdown_interactive(self):
        """Test plotting drawdown (interactive)."""
        # Calculate drawdown first
        self.analyzer.calculate_drawdown(price_column='Close')
        
        # Create plot
        fig = self.analyzer.plot_drawdown(interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_plot_technical_indicators_static(self):
        """Test plotting technical indicators (static)."""
        # Calculate indicators first
        self.analyzer.calculate_moving_averages(price_column='Close')
        self.analyzer.calculate_bollinger_bands(price_column='Close')
        
        # Create plot
        fig = self.analyzer.plot_technical_indicators(price_column='Close', interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_technical_indicators_interactive(self):
        """Test plotting technical indicators (interactive)."""
        # Calculate indicators first
        self.analyzer.calculate_moving_averages(price_column='Close')
        self.analyzer.calculate_bollinger_bands(price_column='Close')
        
        # Create plot
        fig = self.analyzer.plot_technical_indicators(price_column='Close', interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_generate_performance_report(self):
        """Test generating performance report."""
        # Create market data
        market_data = pd.DataFrame({
            'Close': np.linspace(1000, 1200, 100) + np.random.normal(0, 20, 100)
        }, index=self.test_df.index)
        
        # Generate report
        report = self.analyzer.generate_performance_report(benchmark_data=market_data, risk_free_rate=0.02)
        
        # Check that report was generated correctly
        self.assertIsInstance(report, dict)
        self.assertTrue('ticker' in report)
        self.assertTrue('period' in report)
        self.assertTrue('returns' in report)
        self.assertTrue('risk' in report)
        self.assertTrue('risk_adjusted_performance' in report)
        self.assertTrue('benchmark_comparison' in report)

if __name__ == '__main__':
    unittest.main()

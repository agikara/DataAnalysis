"""
Data Analysis Tool - Financial Analysis Module

This module provides functionality for financial data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
from .config import get_config

class FinancialAnalyzer:
    """
    Class for financial data analysis.
    """
    
    def __init__(self, data=None):
        """
        Initialize the FinancialAnalyzer.
        
        Args:
            data (pandas.DataFrame, optional): Financial data to analyze.
        """
        self.data = data
        self.ticker = None
        self.results = {}
    
    def set_data(self, data, ticker=None):
        """
        Set the data to analyze.
        
        Args:
            data (pandas.DataFrame): Financial data to analyze.
            ticker (str, optional): Ticker symbol for the data.
        """
        self.data = data
        self.ticker = ticker
        self.results = {}
    
    def calculate_returns(self, price_column='Close', period='daily'):
        """
        Calculate returns for the financial data.
        
        Args:
            price_column (str, optional): Column containing price data.
            period (str, optional): Return period ('daily', 'weekly', 'monthly', 'annual').
            
        Returns:
            pandas.Series: Returns series.
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        if price_column not in self.data.columns:
            raise ValueError(f"Price column not found: {price_column}")
        
        # Calculate returns based on period
        if period == 'daily':
            returns = self.data[price_column].pct_change()
        elif period == 'weekly':
            returns = self.data[price_column].resample('W').last().pct_change()
        elif period == 'monthly':
            returns = self.data[price_column].resample('M').last().pct_change()
        elif period == 'annual':
            returns = self.data[price_column].resample('Y').last().pct_change()
        else:
            raise ValueError(f"Unknown period: {period}")
        
        # Store results
        self.results[f'{period}_returns'] = returns
        
        return returns
    
    def calculate_cumulative_returns(self, price_column='Close', start_date=None, end_date=None):
        """
        Calculate cumulative returns for the financial data.
        
        Args:
            price_column (str, optional): Column containing price data.
            start_date (str or datetime, optional): Start date for calculation.
            end_date (str or datetime, optional): End date for calculation.
            
        Returns:
            float: Cumulative return.
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        if price_column not in self.data.columns:
            raise ValueError(f"Price column not found: {price_column}")
        
        # Filter data by date if specified
        data = self.data.copy()
        if start_date is not None:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            data = data[data.index <= pd.to_datetime(end_date)]
        
        # Calculate cumulative return
        start_price = data[price_column].iloc[0]
        end_price = data[price_column].iloc[-1]
        cumulative_return = (end_price / start_price) - 1
        
        # Store results
        self.results['cumulative_return'] = cumulative_return
        
        return cumulative_return
    
    def calculate_volatility(self, returns_data=None, period='daily', annualize=True):
        """
        Calculate volatility for the financial data.
        
        Args:
            returns_data (pandas.Series, optional): Returns data. If None, use daily returns.
            period (str, optional): Period of the returns data ('daily', 'weekly', 'monthly').
            annualize (bool, optional): Whether to annualize the volatility.
            
        Returns:
            float: Volatility.
        """
        if returns_data is None:
            # Use daily returns if not specified
            if 'daily_returns' not in self.results:
                self.calculate_returns(period='daily')
            returns_data = self.results['daily_returns']
        
        # Calculate volatility
        volatility = returns_data.std()
        
        # Annualize if requested
        if annualize:
            if period == 'daily':
                volatility *= np.sqrt(252)  # Approx. trading days in a year
            elif period == 'weekly':
                volatility *= np.sqrt(52)   # Weeks in a year
            elif period == 'monthly':
                volatility *= np.sqrt(12)   # Months in a year
        
        # Store results
        self.results['volatility'] = volatility
        
        return volatility
    
    def calculate_sharpe_ratio(self, returns_data=None, risk_free_rate=0.0, period='daily'):
        """
        Calculate Sharpe ratio for the financial data.
        
        Args:
            returns_data (pandas.Series, optional): Returns data. If None, use daily returns.
            risk_free_rate (float, optional): Risk-free rate (annualized).
            period (str, optional): Period of the returns data ('daily', 'weekly', 'monthly').
            
        Returns:
            float: Sharpe ratio.
        """
        if returns_data is None:
            # Use daily returns if not specified
            if 'daily_returns' not in self.results:
                self.calculate_returns(period='daily')
            returns_data = self.results['daily_returns']
        
        # Convert annual risk-free rate to period rate
        if period == 'daily':
            period_risk_free = (1 + risk_free_rate) ** (1/252) - 1
            annualization_factor = np.sqrt(252)
        elif period == 'weekly':
            period_risk_free = (1 + risk_free_rate) ** (1/52) - 1
            annualization_factor = np.sqrt(52)
        elif period == 'monthly':
            period_risk_free = (1 + risk_free_rate) ** (1/12) - 1
            annualization_factor = np.sqrt(12)
        else:
            raise ValueError(f"Unknown period: {period}")
        
        # Calculate Sharpe ratio
        excess_returns = returns_data - period_risk_free
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * annualization_factor
        
        # Store results
        self.results['sharpe_ratio'] = sharpe_ratio
        
        return sharpe_ratio
    
    def calculate_drawdown(self, price_column='Close'):
        """
        Calculate drawdown for the financial data.
        
        Args:
            price_column (str, optional): Column containing price data.
            
        Returns:
            tuple: (Drawdown series, Maximum drawdown, Maximum drawdown duration)
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        if price_column not in self.data.columns:
            raise ValueError(f"Price column not found: {price_column}")
        
        # Calculate drawdown
        prices = self.data[price_column]
        peak = prices.cummax()
        drawdown = (prices / peak) - 1
        
        # Calculate maximum drawdown
        max_drawdown = drawdown.min()
        
        # Calculate maximum drawdown duration
        is_drawdown = drawdown < 0
        if is_drawdown.any():
            # Find the end of the maximum drawdown
            max_drawdown_end = drawdown.idxmin()
            
            # Find the start of the drawdown period (last peak before the maximum drawdown)
            max_drawdown_start = prices[:max_drawdown_end].idxmax()
            
            # Calculate duration
            max_drawdown_duration = max_drawdown_end - max_drawdown_start
        else:
            max_drawdown_duration = pd.Timedelta(0)
        
        # Store results
        self.results['drawdown'] = drawdown
        self.results['max_drawdown'] = max_drawdown
        self.results['max_drawdown_duration'] = max_drawdown_duration
        
        return drawdown, max_drawdown, max_drawdown_duration
    
    def calculate_beta(self, market_data, price_column='Close', market_column='Close'):
        """
        Calculate beta relative to a market index.
        
        Args:
            market_data (pandas.DataFrame): Market index data.
            price_column (str, optional): Column containing price data.
            market_column (str, optional): Column containing market price data.
            
        Returns:
            float: Beta.
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        if price_column not in self.data.columns:
            raise ValueError(f"Price column not found: {price_column}")
        
        if market_column not in market_data.columns:
            raise ValueError(f"Market column not found: {market_column}")
        
        # Calculate returns
        stock_returns = self.data[price_column].pct_change().dropna()
        market_returns = market_data[market_column].pct_change().dropna()
        
        # Align the data
        returns_df = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        })
        returns_df = returns_df.dropna()
        
        # Calculate beta
        covariance = returns_df.cov().loc['stock', 'market']
        market_variance = returns_df['market'].var()
        beta = covariance / market_variance
        
        # Store results
        self.results['beta'] = beta
        
        return beta
    
    def calculate_alpha(self, market_data, risk_free_rate=0.0, price_column='Close', market_column='Close'):
        """
        Calculate Jensen's alpha.
        
        Args:
            market_data (pandas.DataFrame): Market index data.
            risk_free_rate (float, optional): Risk-free rate (annualized).
            price_column (str, optional): Column containing price data.
            market_column (str, optional): Column containing market price data.
            
        Returns:
            float: Alpha.
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        # Calculate beta if not already calculated
        if 'beta' not in self.results:
            beta = self.calculate_beta(market_data, price_column, market_column)
        else:
            beta = self.results['beta']
        
        # Calculate returns
        stock_returns = self.data[price_column].pct_change().dropna()
        market_returns = market_data[market_column].pct_change().dropna()
        
        # Align the data
        returns_df = pd.DataFrame({
            'stock': stock_returns,
            'market': market_returns
        })
        returns_df = returns_df.dropna()
        
        # Convert annual risk-free rate to daily
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        
        # Calculate alpha (annualized)
        stock_return_mean = returns_df['stock'].mean()
        market_return_mean = returns_df['market'].mean()
        alpha_daily = stock_return_mean - (daily_risk_free + beta * (market_return_mean - daily_risk_free))
        alpha_annual = (1 + alpha_daily) ** 252 - 1
        
        # Store results
        self.results['alpha'] = alpha_annual
        
        return alpha_annual
    
    def calculate_moving_averages(self, price_column='Close', windows=[20, 50, 200]):
        """
        Calculate moving averages for the financial data.
        
        Args:
            price_column (str, optional): Column containing price data.
            windows (list, optional): List of window sizes for moving averages.
            
        Returns:
            pandas.DataFrame: DataFrame with moving averages.
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        if price_column not in self.data.columns:
            raise ValueError(f"Price column not found: {price_column}")
        
        # Calculate moving averages
        ma_df = pd.DataFrame(index=self.data.index)
        ma_df[price_column] = self.data[price_column]
        
        for window in windows:
            ma_df[f'MA_{window}'] = self.data[price_column].rolling(window=window).mean()
        
        # Store results
        self.results['moving_averages'] = ma_df
        
        return ma_df
    
    def calculate_rsi(self, price_column='Close', window=14):
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            price_column (str, optional): Column containing price data.
            window (int, optional): Window size for RSI calculation.
            
        Returns:
            pandas.Series: RSI series.
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        if price_column not in self.data.columns:
            raise ValueError(f"Price column not found: {price_column}")
        
        # Calculate price changes
        delta = self.data[price_column].diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Store results
        self.results['rsi'] = rsi
        
        return rsi
    
    def calculate_macd(self, price_column='Close', fast_period=12, slow_period=26, signal_period=9):
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            price_column (str, optional): Column containing price data.
            fast_period (int, optional): Fast EMA period.
            slow_period (int, optional): Slow EMA period.
            signal_period (int, optional): Signal EMA period.
            
        Returns:
            tuple: (MACD line, Signal line, Histogram)
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        if price_column not in self.data.columns:
            raise ValueError(f"Price column not found: {price_column}")
        
        # Calculate EMAs
        fast_ema = self.data[price_column].ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.data[price_column].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        # Store results
        self.results['macd'] = {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, price_column='Close', window=20, num_std=2):
        """
        Calculate Bollinger Bands.
        
        Args:
            price_column (str, optional): Column containing price data.
            window (int, optional): Window size for moving average.
            num_std (int, optional): Number of standard deviations for bands.
            
        Returns:
            tuple: (Middle band, Upper band, Lower band)
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        if price_column not in self.data.columns:
            raise ValueError(f"Price column not found: {price_column}")
        
        # Calculate middle band (SMA)
        middle_band = self.data[price_column].rolling(window=window).mean()
        
        # Calculate standard deviation
        std = self.data[price_column].rolling(window=window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        # Store results
        self.results['bollinger_bands'] = {
            'middle_band': middle_band,
            'upper_band': upper_band,
            'lower_band': lower_band
        }
        
        return middle_band, upper_band, lower_band
    
    def calculate_performance_metrics(self, benchmark_data=None, risk_free_rate=0.0):
        """
        Calculate comprehensive performance metrics.
        
        Args:
            benchmark_data (pandas.DataFrame, optional): Benchmark data for comparison.
            risk_free_rate (float, optional): Risk-free rate (annualized).
            
        Returns:
            dict: Performance metrics.
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        metrics = {}
        
        # Calculate returns if not already calculated
        if 'daily_returns' not in self.results:
            daily_returns = self.calculate_returns(period='daily')
        else:
            daily_returns = self.results['daily_returns']
        
        # Calculate cumulative return if not already calculated
        if 'cumulative_return' not in self.results:
            cumulative_return = self.calculate_cumulative_returns()
        else:
            cumulative_return = self.results['cumulative_return']
        
        # Calculate annualized return
        total_days = (self.data.index[-1] - self.data.index[0]).days
        annualized_return = (1 + cumulative_return) ** (365 / total_days) - 1
        
        # Calculate volatility if not already calculated
        if 'volatility' not in self.results:
            volatility = self.calculate_volatility()
        else:
            volatility = self.results['volatility']
        
        # Calculate Sharpe ratio if not already calculated
        if 'sharpe_ratio' not in self.results:
            sharpe_ratio = self.calculate_sharpe_ratio(risk_free_rate=risk_free_rate)
        else:
            sharpe_ratio = self.results['sharpe_ratio']
        
        # Calculate drawdown if not already calculated
        if 'max_drawdown' not in self.results:
            _, max_drawdown, max_drawdown_duration = self.calculate_drawdown()
        else:
            max_drawdown = self.results['max_drawdown']
            max_drawdown_duration = self.results['max_drawdown_duration']
        
        # Calculate beta and alpha if benchmark data is provided
        if benchmark_data is not None:
            if 'beta' not in self.results:
                beta = self.calculate_beta(benchmark_data)
            else:
                beta = self.results['beta']
            
            if 'alpha' not in self.results:
                alpha = self.calculate_alpha(benchmark_data, risk_free_rate)
            else:
                alpha = self.results['alpha']
            
            metrics['beta'] = beta
            metrics['alpha'] = alpha
        
        # Compile metrics
        metrics.update({
            'ticker': self.ticker,
            'start_date': self.data.index[0].strftime('%Y-%m-%d'),
            'end_date': self.data.index[-1].strftime('%Y-%m-%d'),
            'total_days': total_days,
            'cumulative_return': cumulative_return,
            'annualized_return': annualized_return,
            'annualized_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration.days,
            'risk_free_rate': risk_free_rate
        })
        
        # Store results
        self.results['performance_metrics'] = metrics
        
        return metrics
    
    def plot_price_history(self, price_column='Close', title=None, figsize=None, interactive=True):
        """
        Plot price history.
        
        Args:
            price_column (str, optional): Column containing price data.
            title (str, optional): Plot title.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        if price_column not in self.data.columns:
            raise ValueError(f"Price column not found: {price_column}")
        
        if title is None:
            title = f"{self.ticker} Price History" if self.ticker else "Price History"
        
        if interactive:
            # Create interactive plot with Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data[price_column],
                mode='lines',
                name=price_column
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified'
            )
            
            return fig
        else:
            # Create static plot with Matplotlib
            if figsize is None:
                figsize = (10, 6)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.plot(self.data.index, self.data[price_column])
            
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
    
    def plot_returns_distribution(self, returns_data=None, title=None, figsize=None, interactive=True):
        """
        Plot returns distribution.
        
        Args:
            returns_data (pandas.Series, optional): Returns data. If None, use daily returns.
            title (str, optional): Plot title.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            
        Returns:
            The plot object.
        """
        if returns_data is None:
            # Use daily returns if not specified
            if 'daily_returns' not in self.results:
                self.calculate_returns(period='daily')
            returns_data = self.results['daily_returns']
        
        if title is None:
            title = f"{self.ticker} Returns Distribution" if self.ticker else "Returns Distribution"
        
        if interactive:
            # Create interactive plot with Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=returns_data.dropna(),
                nbinsx=50,
                name='Returns'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Return',
                yaxis_title='Frequency',
                bargap=0.1
            )
            
            return fig
        else:
            # Create static plot with Matplotlib
            if figsize is None:
                figsize = (10, 6)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.hist(returns_data.dropna(), bins=50)
            
            ax.set_title(title)
            ax.set_xlabel('Return')
            ax.set_ylabel('Frequency')
            
            plt.tight_layout()
            
            return fig
    
    def plot_cumulative_returns(self, returns_data=None, title=None, figsize=None, interactive=True):
        """
        Plot cumulative returns.
        
        Args:
            returns_data (pandas.Series, optional): Returns data. If None, use daily returns.
            title (str, optional): Plot title.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            
        Returns:
            The plot object.
        """
        if returns_data is None:
            # Use daily returns if not specified
            if 'daily_returns' not in self.results:
                self.calculate_returns(period='daily')
            returns_data = self.results['daily_returns']
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns_data).cumprod() - 1
        
        if title is None:
            title = f"{self.ticker} Cumulative Returns" if self.ticker else "Cumulative Returns"
        
        if interactive:
            # Create interactive plot with Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Cumulative Return',
                hovermode='x unified'
            )
            
            return fig
        else:
            # Create static plot with Matplotlib
            if figsize is None:
                figsize = (10, 6)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.plot(cumulative_returns.index, cumulative_returns)
            
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Cumulative Return')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
    
    def plot_drawdown(self, title=None, figsize=None, interactive=True):
        """
        Plot drawdown.
        
        Args:
            title (str, optional): Plot title.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            
        Returns:
            The plot object.
        """
        # Calculate drawdown if not already calculated
        if 'drawdown' not in self.results:
            drawdown, _, _ = self.calculate_drawdown()
        else:
            drawdown = self.results['drawdown']
        
        if title is None:
            title = f"{self.ticker} Drawdown" if self.ticker else "Drawdown"
        
        if interactive:
            # Create interactive plot with Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.3)'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Drawdown',
                hovermode='x unified'
            )
            
            return fig
        else:
            # Create static plot with Matplotlib
            if figsize is None:
                figsize = (10, 6)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
            ax.plot(drawdown.index, drawdown, color='red')
            
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
    
    def plot_technical_indicators(self, price_column='Close', title=None, figsize=None, interactive=True):
        """
        Plot technical indicators.
        
        Args:
            price_column (str, optional): Column containing price data.
            title (str, optional): Plot title.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        # Calculate indicators if not already calculated
        if 'moving_averages' not in self.results:
            self.calculate_moving_averages(price_column=price_column)
        
        if 'bollinger_bands' not in self.results:
            self.calculate_bollinger_bands(price_column=price_column)
        
        # Get data
        ma_df = self.results['moving_averages']
        bb = self.results['bollinger_bands']
        
        if title is None:
            title = f"{self.ticker} Technical Indicators" if self.ticker else "Technical Indicators"
        
        if interactive:
            # Create interactive plot with Plotly
            fig = go.Figure()
            
            # Add price
            fig.add_trace(go.Scatter(
                x=self.data.index,
                y=self.data[price_column],
                mode='lines',
                name=price_column
            ))
            
            # Add moving averages
            for col in ma_df.columns:
                if col != price_column:
                    fig.add_trace(go.Scatter(
                        x=ma_df.index,
                        y=ma_df[col],
                        mode='lines',
                        name=col
                    ))
            
            # Add Bollinger Bands
            fig.add_trace(go.Scatter(
                x=bb['upper_band'].index,
                y=bb['upper_band'],
                mode='lines',
                name='Upper Band',
                line=dict(dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=bb['lower_band'].index,
                y=bb['lower_band'],
                mode='lines',
                name='Lower Band',
                line=dict(dash='dash')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified'
            )
            
            return fig
        else:
            # Create static plot with Matplotlib
            if figsize is None:
                figsize = (10, 6)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot price
            ax.plot(self.data.index, self.data[price_column], label=price_column)
            
            # Plot moving averages
            for col in ma_df.columns:
                if col != price_column:
                    ax.plot(ma_df.index, ma_df[col], label=col)
            
            # Plot Bollinger Bands
            ax.plot(bb['upper_band'].index, bb['upper_band'], 'k--', label='Upper Band')
            ax.plot(bb['lower_band'].index, bb['lower_band'], 'k--', label='Lower Band')
            
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return fig
    
    def generate_performance_report(self, benchmark_data=None, risk_free_rate=0.0):
        """
        Generate a comprehensive performance report.
        
        Args:
            benchmark_data (pandas.DataFrame, optional): Benchmark data for comparison.
            risk_free_rate (float, optional): Risk-free rate (annualized).
            
        Returns:
            dict: Performance report.
        """
        if self.data is None:
            raise ValueError("No data to analyze")
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(benchmark_data, risk_free_rate)
        
        # Format the report
        report = {
            'ticker': metrics['ticker'],
            'period': f"{metrics['start_date']} to {metrics['end_date']} ({metrics['total_days']} days)",
            'returns': {
                'cumulative_return': f"{metrics['cumulative_return']:.2%}",
                'annualized_return': f"{metrics['annualized_return']:.2%}",
            },
            'risk': {
                'annualized_volatility': f"{metrics['annualized_volatility']:.2%}",
                'max_drawdown': f"{metrics['max_drawdown']:.2%}",
                'max_drawdown_duration': f"{metrics['max_drawdown_duration']} days",
            },
            'risk_adjusted_performance': {
                'sharpe_ratio': f"{metrics['sharpe_ratio']:.2f}",
            }
        }
        
        # Add benchmark-related metrics if available
        if 'beta' in metrics:
            report['benchmark_comparison'] = {
                'beta': f"{metrics['beta']:.2f}",
                'alpha': f"{metrics['alpha']:.2%}",
            }
        
        return report

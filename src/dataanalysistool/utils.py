"""
Data Analysis Tool - Utils Module

This module provides utility functions for the Data Analysis Tool.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import json
import re
from .config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dataanalysistool')

def setup_logger(level=None):
    """
    Set up logger with specified level.
    
    Args:
        level (str, optional): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
    """
    if level is None:
        level = get_config('log_level')
    
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logger.setLevel(numeric_level)

def detect_date_format(date_string):
    """
    Detect the format of a date string.
    
    Args:
        date_string (str): Date string to analyze.
        
    Returns:
        str: Detected date format.
    """
    formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',
        '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y',
        '%Y%m%d', '%d%m%Y', '%m%d%Y',
        '%b %d, %Y', '%B %d, %Y',
        '%d %b %Y', '%d %B %Y'
    ]
    
    for fmt in formats:
        try:
            datetime.strptime(date_string, fmt)
            return fmt
        except ValueError:
            continue
    
    return None

def infer_column_types(df):
    """
    Infer column types for a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame to analyze.
        
    Returns:
        dict: Dictionary mapping column names to inferred types.
    """
    column_types = {}
    
    for column in df.columns:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            if pd.api.types.is_integer_dtype(df[column]):
                column_types[column] = 'integer'
            else:
                column_types[column] = 'float'
        
        # Check if column is datetime
        elif pd.api.types.is_datetime64_dtype(df[column]):
            column_types[column] = 'datetime'
        
        # Check if column could be datetime
        elif df[column].dtype == 'object':
            # Sample non-null values
            sample = df[column].dropna().head(10)
            if len(sample) > 0:
                # Try to detect date format
                date_format = detect_date_format(sample.iloc[0])
                if date_format:
                    try:
                        pd.to_datetime(sample, format=date_format)
                        column_types[column] = 'datetime'
                        continue
                    except:
                        pass
            
            # Check if column is categorical
            if df[column].nunique() < 0.2 * len(df):
                column_types[column] = 'categorical'
            else:
                column_types[column] = 'text'
    
    return column_types

def detect_outliers(series, method='zscore', threshold=3):
    """
    Detect outliers in a series.
    
    Args:
        series (pandas.Series): Series to analyze.
        method (str, optional): Method for outlier detection ('zscore', 'iqr').
        threshold (float, optional): Threshold for outlier detection.
        
    Returns:
        pandas.Series: Boolean series indicating outliers.
    """
    if method == 'zscore':
        # Z-score method
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    elif method == 'iqr':
        # IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

def create_sample_data(n_rows=1000, seed=None):
    """
    Create sample data for testing.
    
    Args:
        n_rows (int, optional): Number of rows to generate.
        seed (int, optional): Random seed.
        
    Returns:
        pandas.DataFrame: Sample data.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Date range
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start=start_date, periods=n_rows, freq='D')
    
    # Generate data
    df = pd.DataFrame({
        'date': dates,
        'value': np.random.normal(100, 15, n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
        'quantity': np.random.randint(1, 100, n_rows),
        'is_active': np.random.choice([True, False], n_rows),
        'score': np.random.uniform(0, 1, n_rows)
    })
    
    # Add some missing values
    mask = np.random.random(n_rows) < 0.05
    df.loc[mask, 'value'] = np.nan
    
    # Add some outliers
    outlier_mask = np.random.random(n_rows) < 0.02
    df.loc[outlier_mask, 'value'] = np.random.normal(200, 30, outlier_mask.sum())
    
    return df

def create_sample_stock_data(ticker='SAMPLE', n_days=252, seed=None):
    """
    Create sample stock data for testing.
    
    Args:
        ticker (str, optional): Ticker symbol.
        n_days (int, optional): Number of days to generate.
        seed (int, optional): Random seed.
        
    Returns:
        pandas.DataFrame: Sample stock data.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Date range (trading days)
    end_date = datetime.now()
    dates = pd.date_range(end=end_date, periods=n_days, freq='B')
    
    # Initial price
    price = 100.0
    
    # Generate price data with random walk
    prices = [price]
    for _ in range(1, n_days):
        change_percent = np.random.normal(0.0005, 0.015)
        price *= (1 + change_percent)
        prices.append(price)
    
    prices = np.array(prices)
    
    # Generate OHLC data
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'Close': prices,
        'Volume': np.random.randint(100000, 10000000, n_days)
    }, index=dates)
    
    # Ensure High is the highest and Low is the lowest
    for i in range(n_days):
        high = max(df.iloc[i]['Open'], df.iloc[i]['Close'], df.iloc[i]['High'])
        low = min(df.iloc[i]['Open'], df.iloc[i]['Close'], df.iloc[i]['Low'])
        df.iloc[i, df.columns.get_loc('High')] = high
        df.iloc[i, df.columns.get_loc('Low')] = low
    
    # Add Adjusted Close
    df['Adj Close'] = df['Close']
    
    return df

def validate_ticker_symbol(ticker):
    """
    Validate a ticker symbol.
    
    Args:
        ticker (str): Ticker symbol to validate.
        
    Returns:
        bool: Whether the ticker symbol is valid.
    """
    # Basic validation: 1-5 uppercase letters
    pattern = re.compile(r'^[A-Z]{1,5}$')
    return bool(pattern.match(ticker))

def format_number(number, precision=2, include_commas=True, as_percent=False):
    """
    Format a number for display.
    
    Args:
        number (float): Number to format.
        precision (int, optional): Decimal precision.
        include_commas (bool, optional): Whether to include commas as thousands separators.
        as_percent (bool, optional): Whether to format as percentage.
        
    Returns:
        str: Formatted number.
    """
    if pd.isna(number):
        return 'N/A'
    
    if as_percent:
        number *= 100
        format_str = f'{{:.{precision}f}}%'
        return format_str.format(number)
    
    if include_commas:
        format_str = f'{{:,.{precision}f}}'
    else:
        format_str = f'{{:.{precision}f}}'
    
    return format_str.format(number)

def save_config(config, filename):
    """
    Save configuration to a file.
    
    Args:
        config (dict): Configuration to save.
        filename (str): Filename to save to.
        
    Returns:
        str: Path to the saved file.
    """
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    return filename

def load_config(filename):
    """
    Load configuration from a file.
    
    Args:
        filename (str): Filename to load from.
        
    Returns:
        dict: Loaded configuration.
    """
    with open(filename, 'r') as f:
        config = json.load(f)
    
    return config

def set_plot_style(style=None, context=None, palette=None, font_scale=None):
    """
    Set the plot style for matplotlib and seaborn.
    
    Args:
        style (str, optional): Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks').
        context (str, optional): Seaborn context ('paper', 'notebook', 'talk', 'poster').
        palette (str, optional): Color palette.
        font_scale (float, optional): Font scale.
    """
    if style is None:
        style = get_config('default_theme')
    
    if palette is None:
        palette = get_config('color_palette')
    
    # Set seaborn style
    sns.set_style(style)
    
    if context:
        sns.set_context(context, font_scale=font_scale)
    
    if palette:
        sns.set_palette(palette)
    
    # Set matplotlib defaults
    plt.rcParams['figure.figsize'] = get_config('default_figsize')
    plt.rcParams['figure.dpi'] = get_config('default_dpi')

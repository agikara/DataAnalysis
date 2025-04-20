"""
Data Analysis Tool - Data Loader Module

This module provides functionality for loading data from various sources.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
import sys
from datetime import datetime
from .config import get_config

# Try to import the data API client for Yahoo Finance API
try:
    sys.path.append('/opt/.manus/.sandbox-runtime')
    from data_api import ApiClient
    HAS_DATA_API = True
except ImportError:
    HAS_DATA_API = False


class DataLoader:
    """
    Class for loading data from various sources.
    """
    
    def __init__(self):
        """Initialize the DataLoader."""
        self.data = None
        self.source = None
        self.metadata = {}
    
    def load_csv(self, filepath, **kwargs):
        """
        Load data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file.
            **kwargs: Additional arguments to pass to pandas.read_csv.
            
        Returns:
            pandas.DataFrame: The loaded data.
        """
        encoding = kwargs.pop('encoding', get_config('csv_encoding'))
        self.data = pd.read_csv(filepath, encoding=encoding, **kwargs)
        self.source = f"CSV: {os.path.basename(filepath)}"
        self.metadata = {
            'source_type': 'csv',
            'filepath': filepath,
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return self.data
    
    def load_excel(self, filepath, sheet_name=0, **kwargs):
        """
        Load data from an Excel file.
        
        Args:
            filepath (str): Path to the Excel file.
            sheet_name (str or int, optional): Sheet to load. Defaults to 0.
            **kwargs: Additional arguments to pass to pandas.read_excel.
            
        Returns:
            pandas.DataFrame: The loaded data.
        """
        engine = kwargs.pop('engine', get_config('excel_engine'))
        self.data = pd.read_excel(filepath, sheet_name=sheet_name, engine=engine, **kwargs)
        self.source = f"Excel: {os.path.basename(filepath)}, Sheet: {sheet_name}"
        self.metadata = {
            'source_type': 'excel',
            'filepath': filepath,
            'sheet_name': sheet_name,
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return self.data
    
    def load_json(self, filepath, **kwargs):
        """
        Load data from a JSON file.
        
        Args:
            filepath (str): Path to the JSON file.
            **kwargs: Additional arguments to pass to pandas.read_json.
            
        Returns:
            pandas.DataFrame: The loaded data.
        """
        self.data = pd.read_json(filepath, **kwargs)
        self.source = f"JSON: {os.path.basename(filepath)}"
        self.metadata = {
            'source_type': 'json',
            'filepath': filepath,
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return self.data
    
    def load_sql(self, query, connection, **kwargs):
        """
        Load data from a SQL database.
        
        Args:
            query (str): SQL query to execute.
            connection: SQLAlchemy connectable or connection string.
            **kwargs: Additional arguments to pass to pandas.read_sql.
            
        Returns:
            pandas.DataFrame: The loaded data.
        """
        self.data = pd.read_sql(query, connection, **kwargs)
        self.source = f"SQL: {query[:50]}..." if len(query) > 50 else f"SQL: {query}"
        self.metadata = {
            'source_type': 'sql',
            'query': query,
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return self.data
    
    def load_stock_data(self, ticker, period=None, start=None, end=None, interval='1d'):
        """
        Load stock data from Yahoo Finance.
        
        Args:
            ticker (str): Stock ticker symbol.
            period (str, optional): Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max).
            start (str, optional): Start date in YYYY-MM-DD format.
            end (str, optional): End date in YYYY-MM-DD format.
            interval (str, optional): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo).
            
        Returns:
            pandas.DataFrame: The loaded stock data.
        """
        # Use default period if not specified
        if period is None and start is None and end is None:
            period = get_config('default_period')
        
        # Load data using yfinance
        if period is not None:
            self.data = yf.download(ticker, period=period, interval=interval)
        else:
            self.data = yf.download(ticker, start=start, end=end, interval=interval)
        
        self.source = f"Yahoo Finance: {ticker}"
        self.metadata = {
            'source_type': 'yahoo_finance',
            'ticker': ticker,
            'period': period,
            'start': start,
            'end': end,
            'interval': interval,
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return self.data
    
    def load_stock_data_api(self, symbol, interval='1mo', range='1mo', region='US'):
        """
        Load stock data using the Yahoo Finance API.
        
        Args:
            symbol (str): Stock ticker symbol.
            interval (str, optional): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo).
            range (str, optional): Data range (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max).
            region (str, optional): Region code (US, BR, AU, CA, FR, DE, HK, IN, IT, ES, GB, SG).
            
        Returns:
            pandas.DataFrame: The loaded stock data.
        """
        if not HAS_DATA_API:
            raise ImportError("Data API client not available. Falling back to yfinance.")
        
        # Use the data API client
        client = ApiClient()
        result = client.call_api('YahooFinance/get_stock_chart', query={
            'symbol': symbol,
            'interval': interval,
            'range': range,
            'region': region,
            'includeAdjustedClose': True
        })
        
        # Process the API response
        if result and 'chart' in result and 'result' in result['chart'] and len(result['chart']['result']) > 0:
            chart_data = result['chart']['result'][0]
            
            # Extract timestamp and indicators
            timestamps = chart_data.get('timestamp', [])
            indicators = chart_data.get('indicators', {})
            quote = indicators.get('quote', [{}])[0] if 'quote' in indicators else {}
            
            # Create DataFrame
            data = {
                'Open': quote.get('open', []),
                'High': quote.get('high', []),
                'Low': quote.get('low', []),
                'Close': quote.get('close', []),
                'Volume': quote.get('volume', [])
            }
            
            # Add adjusted close if available
            if 'adjclose' in indicators:
                adjclose = indicators['adjclose'][0] if len(indicators['adjclose']) > 0 else {}
                data['Adj Close'] = adjclose.get('adjclose', [])
            
            # Create DataFrame with timestamp as index
            df = pd.DataFrame(data)
            if timestamps:
                df.index = pd.to_datetime(timestamps, unit='s')
                df.index.name = 'Date'
            
            self.data = df
            self.source = f"Yahoo Finance API: {symbol}"
            self.metadata = {
                'source_type': 'yahoo_finance_api',
                'symbol': symbol,
                'interval': interval,
                'range': range,
                'region': region,
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            return self.data
        else:
            raise ValueError(f"Failed to retrieve data for {symbol}")
    
    def load_stock_insights(self, symbol):
        """
        Load stock insights using the Yahoo Finance API.
        
        Args:
            symbol (str): Stock ticker symbol.
            
        Returns:
            dict: Stock insights data.
        """
        if not HAS_DATA_API:
            raise ImportError("Data API client not available.")
        
        # Use the data API client
        client = ApiClient()
        result = client.call_api('YahooFinance/get_stock_insights', query={
            'symbol': symbol
        })
        
        # Process the API response
        if result and 'finance' in result and 'result' in result['finance']:
            insights = result['finance']['result']
            
            # Store insights in metadata
            self.metadata['insights'] = insights
            
            return insights
        else:
            raise ValueError(f"Failed to retrieve insights for {symbol}")
    
    def get_metadata(self):
        """
        Get metadata about the loaded data.
        
        Returns:
            dict: Metadata about the loaded data.
        """
        return self.metadata
    
    def get_data_summary(self):
        """
        Get a summary of the loaded data.
        
        Returns:
            dict: Summary of the loaded data.
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        summary = {
            "source": self.source,
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
            "missing_values": self.data.isna().sum().to_dict(),
            "numeric_columns": list(self.data.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.data.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": list(self.data.select_dtypes(include=['datetime']).columns),
        }
        
        return summary

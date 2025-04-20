"""
Data Analysis Tool - Tests for Data Loader Module

This module contains tests for the DataLoader class.
"""

import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module to test
from dataanalysistool.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
        
        # Create test data directory
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create test CSV file
        self.csv_path = os.path.join(self.test_dir, 'test.csv')
        self.test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.1, 2.2, 3.3]
        })
        self.test_df.to_csv(self.csv_path, index=False)
        
        # Create test Excel file
        self.excel_path = os.path.join(self.test_dir, 'test.xlsx')
        self.test_df.to_excel(self.excel_path, index=False)
        
        # Create test JSON file
        self.json_path = os.path.join(self.test_dir, 'test.json')
        self.test_df.to_json(self.json_path, orient='records')
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove test files
        for file_path in [self.csv_path, self.excel_path, self.json_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove test directory
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    def test_load_csv(self):
        """Test loading CSV file."""
        df = self.loader.load_csv(self.csv_path)
        
        # Check that the DataFrame was loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (3, 3))
        self.assertEqual(list(df.columns), ['A', 'B', 'C'])
        self.assertEqual(df['A'].tolist(), [1, 2, 3])
    
    def test_load_excel(self):
        """Test loading Excel file."""
        df = self.loader.load_excel(self.excel_path)
        
        # Check that the DataFrame was loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (3, 3))
        self.assertEqual(list(df.columns), ['A', 'B', 'C'])
        self.assertEqual(df['A'].tolist(), [1, 2, 3])
    
    def test_load_json(self):
        """Test loading JSON file."""
        df = self.loader.load_json(self.json_path)
        
        # Check that the DataFrame was loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (3, 3))
        self.assertEqual(sorted(list(df.columns)), ['A', 'B', 'C'])
        self.assertEqual(df['A'].tolist(), [1, 2, 3])
    
    def test_load_sql(self):
        """Test loading data from SQL."""
        # Mock the pandas read_sql function
        with patch('pandas.read_sql') as mock_read_sql:
            mock_read_sql.return_value = self.test_df
            
            # Mock the database connection
            mock_conn = MagicMock()
            
            # Call the method
            df = self.loader.load_sql('SELECT * FROM test', mock_conn)
            
            # Check that read_sql was called with the correct arguments
            mock_read_sql.assert_called_once_with('SELECT * FROM test', mock_conn)
            
            # Check that the DataFrame was loaded correctly
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(df.shape, (3, 3))
            self.assertEqual(list(df.columns), ['A', 'B', 'C'])
            self.assertEqual(df['A'].tolist(), [1, 2, 3])
    
    @patch('yfinance.download')
    def test_load_stock_data(self, mock_download):
        """Test loading stock data."""
        # Create mock stock data
        index = pd.date_range(start='2020-01-01', periods=5, freq='D')
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1000, 1100, 1200, 1300, 1400],
            'Adj Close': [102, 103, 104, 105, 106]
        }, index=index)
        
        # Set up the mock
        mock_download.return_value = mock_data
        
        # Call the method
        df = self.loader.load_stock_data('AAPL', period='1mo', interval='1d')
        
        # Check that download was called with the correct arguments
        mock_download.assert_called_once_with(tickers='AAPL', period='1mo', interval='1d')
        
        # Check that the DataFrame was loaded correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (5, 6))
        self.assertEqual(list(df.columns), ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'])
        self.assertEqual(df['Close'].tolist(), [102, 103, 104, 105, 106])

if __name__ == '__main__':
    unittest.main()

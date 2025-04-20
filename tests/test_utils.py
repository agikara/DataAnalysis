"""
Data Analysis Tool - Tests for Utils Module

This module contains tests for the utility functions.
"""

import os
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from unittest.mock import patch, MagicMock

# Import the module to test
from dataanalysistool.utils import (
    setup_logger, detect_date_format, infer_column_types, detect_outliers,
    create_sample_data, create_sample_stock_data, validate_ticker_symbol,
    format_number, save_config, load_config, set_plot_style
)

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test data
        self.test_df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'value': [1, 2, 30],  # 30 is an outlier
            'category': ['A', 'B', 'A'],
            'price': [10.5, 11.2, 9.8]
        })
        
        # Create test directory
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_utils')
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """Tear down test fixtures."""
        plt.close('all')  # Close all matplotlib figures
        
        # Remove test files
        if os.path.exists(os.path.join(self.test_dir, 'test_config.json')):
            os.remove(os.path.join(self.test_dir, 'test_config.json'))
        
        # Remove test directory
        if os.path.exists(self.test_dir):
            os.rmdir(self.test_dir)
    
    @patch('logging.Logger.setLevel')
    def test_setup_logger(self, mock_set_level):
        """Test setting up logger."""
        # Set up logger with INFO level
        setup_logger(level='INFO')
        
        # Check that setLevel was called with INFO level
        mock_set_level.assert_called_once_with(logging.INFO)
    
    def test_detect_date_format(self):
        """Test detecting date format."""
        # Test various date formats
        self.assertEqual(detect_date_format('2020-01-01'), '%Y-%m-%d')
        self.assertEqual(detect_date_format('01/01/2020'), '%m/%d/%Y')
        self.assertEqual(detect_date_format('Jan 1, 2020'), '%b %d, %Y')
        self.assertIsNone(detect_date_format('not a date'))
    
    def test_infer_column_types(self):
        """Test inferring column types."""
        # Infer column types
        column_types = infer_column_types(self.test_df)
        
        # Check that types were inferred correctly
        self.assertEqual(column_types['value'], 'integer')
        self.assertEqual(column_types['category'], 'categorical')
        self.assertEqual(column_types['price'], 'float')
        # 'date' could be inferred as datetime or text depending on implementation
    
    def test_detect_outliers_zscore(self):
        """Test detecting outliers using Z-score method."""
        # Detect outliers
        outliers = detect_outliers(self.test_df['value'], method='zscore', threshold=2)
        
        # Check that outliers were detected correctly
        self.assertEqual(outliers.sum(), 1)  # One outlier
        self.assertTrue(outliers.iloc[2])  # Third value is an outlier
    
    def test_detect_outliers_iqr(self):
        """Test detecting outliers using IQR method."""
        # Detect outliers
        outliers = detect_outliers(self.test_df['value'], method='iqr', threshold=1.5)
        
        # Check that outliers were detected correctly
        self.assertEqual(outliers.sum(), 1)  # One outlier
        self.assertTrue(outliers.iloc[2])  # Third value is an outlier
    
    def test_create_sample_data(self):
        """Test creating sample data."""
        # Create sample data
        df = create_sample_data(n_rows=100, seed=42)
        
        # Check that data was created correctly
        self.assertEqual(df.shape[0], 100)
        self.assertTrue('date' in df.columns)
        self.assertTrue('value' in df.columns)
        self.assertTrue('category' in df.columns)
        self.assertTrue('quantity' in df.columns)
        self.assertTrue('is_active' in df.columns)
        self.assertTrue('score' in df.columns)
    
    def test_create_sample_stock_data(self):
        """Test creating sample stock data."""
        # Create sample stock data
        df = create_sample_stock_data(ticker='TEST', n_days=100, seed=42)
        
        # Check that data was created correctly
        self.assertEqual(df.shape[0], 100)
        self.assertTrue('Open' in df.columns)
        self.assertTrue('High' in df.columns)
        self.assertTrue('Low' in df.columns)
        self.assertTrue('Close' in df.columns)
        self.assertTrue('Volume' in df.columns)
        self.assertTrue('Adj Close' in df.columns)
        
        # Check that High is always >= Open and Close
        self.assertTrue(all(df['High'] >= df['Open']))
        self.assertTrue(all(df['High'] >= df['Close']))
        
        # Check that Low is always <= Open and Close
        self.assertTrue(all(df['Low'] <= df['Open']))
        self.assertTrue(all(df['Low'] <= df['Close']))
    
    def test_validate_ticker_symbol(self):
        """Test validating ticker symbols."""
        # Test valid ticker symbols
        self.assertTrue(validate_ticker_symbol('AAPL'))
        self.assertTrue(validate_ticker_symbol('MSFT'))
        self.assertTrue(validate_ticker_symbol('GOOG'))
        
        # Test invalid ticker symbols
        self.assertFalse(validate_ticker_symbol('aapl'))  # Lowercase
        self.assertFalse(validate_ticker_symbol('AAPL1'))  # Contains number
        self.assertFalse(validate_ticker_symbol('APPLE'))  # Too long
        self.assertFalse(validate_ticker_symbol(''))  # Empty
    
    def test_format_number(self):
        """Test formatting numbers."""
        # Test formatting numbers
        self.assertEqual(format_number(1234.5678), '1,234.57')
        self.assertEqual(format_number(1234.5678, precision=3), '1,234.568')
        self.assertEqual(format_number(1234.5678, include_commas=False), '1234.57')
        self.assertEqual(format_number(0.1234, as_percent=True), '12.34%')
        self.assertEqual(format_number(np.nan), 'N/A')
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Create test config
        config = {
            'key1': 'value1',
            'key2': 123,
            'key3': [1, 2, 3],
            'key4': {'nested': 'value'}
        }
        
        # Save config
        filepath = os.path.join(self.test_dir, 'test_config.json')
        save_config(config, filepath)
        
        # Check that file was created
        self.assertTrue(os.path.exists(filepath))
        
        # Load config
        loaded_config = load_config(filepath)
        
        # Check that config was loaded correctly
        self.assertEqual(loaded_config, config)
    
    @patch('seaborn.set_style')
    @patch('seaborn.set_context')
    @patch('seaborn.set_palette')
    def test_set_plot_style(self, mock_set_palette, mock_set_context, mock_set_style):
        """Test setting plot style."""
        # Set plot style
        set_plot_style(style='darkgrid', context='notebook', palette='deep', font_scale=1.2)
        
        # Check that style functions were called
        mock_set_style.assert_called_once_with('darkgrid')
        mock_set_context.assert_called_once_with('notebook', font_scale=1.2)
        mock_set_palette.assert_called_once_with('deep')

if __name__ == '__main__':
    unittest.main()

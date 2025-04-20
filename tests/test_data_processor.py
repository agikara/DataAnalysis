"""
Data Analysis Tool - Tests for Data Processor Module

This module contains tests for the DataProcessor class.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

# Import the module to test
from dataanalysistool.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DataProcessor()
        
        # Create test data
        self.test_df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': ['a', 'b', 'c', None, 'e'],
            'C': [1.1, 2.2, 3.3, 4.4, 100.0],  # 100.0 is an outlier
            'D': [True, False, True, False, True]
        })
        
        # Set data for processor
        self.processor.set_data(self.test_df)
    
    def test_get_data(self):
        """Test getting data."""
        df = self.processor.get_data()
        
        # Check that the DataFrame was returned correctly
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (5, 4))
        self.assertEqual(list(df.columns), ['A', 'B', 'C', 'D'])
    
    def test_describe(self):
        """Test describing data."""
        desc = self.processor.describe()
        
        # Check that the description was generated correctly
        self.assertIsInstance(desc, pd.DataFrame)
        self.assertEqual(desc.shape[1], 2)  # Only numeric columns A and C
        self.assertTrue('A' in desc.columns)
        self.assertTrue('C' in desc.columns)
    
    def test_handle_missing_values_mean(self):
        """Test handling missing values with mean strategy."""
        self.processor.handle_missing_values(strategy='mean')
        df = self.processor.get_data()
        
        # Check that missing values were filled correctly
        self.assertFalse(df['A'].isna().any())
        self.assertTrue(df['B'].isna().any())  # Non-numeric column should still have NaN
        self.assertEqual(df['A'][2], 3.0)  # Mean of [1, 2, 4, 5]
    
    def test_handle_missing_values_median(self):
        """Test handling missing values with median strategy."""
        self.processor.handle_missing_values(strategy='median')
        df = self.processor.get_data()
        
        # Check that missing values were filled correctly
        self.assertFalse(df['A'].isna().any())
        self.assertTrue(df['B'].isna().any())  # Non-numeric column should still have NaN
        self.assertEqual(df['A'][2], 3.0)  # Median of [1, 2, 4, 5]
    
    def test_handle_missing_values_mode(self):
        """Test handling missing values with mode strategy."""
        self.processor.handle_missing_values(strategy='mode')
        df = self.processor.get_data()
        
        # Check that missing values were filled correctly
        self.assertFalse(df['A'].isna().any())
        self.assertFalse(df['B'].isna().any())
        # Mode of [1, 2, 4, 5] is 1 (first value in case of tie)
        self.assertEqual(df['A'][2], 1.0)
        # Mode of ['a', 'b', 'c', 'e'] is 'a' (first value in case of tie)
        self.assertEqual(df['B'][3], 'a')
    
    def test_normalize(self):
        """Test normalizing data."""
        self.processor.normalize()
        df = self.processor.get_data()
        
        # Check that numeric columns were normalized correctly
        self.assertTrue(df['A'].min() >= 0)
        self.assertTrue(df['A'].max() <= 1)
        self.assertTrue(df['C'].min() >= 0)
        self.assertTrue(df['C'].max() <= 1)
        
        # Check that non-numeric columns were not modified
        self.assertEqual(df['B'].tolist(), ['a', 'b', 'c', None, 'e'])
        self.assertEqual(df['D'].tolist(), [True, False, True, False, True])
    
    def test_remove_outliers(self):
        """Test removing outliers."""
        self.processor.remove_outliers()
        df = self.processor.get_data()
        
        # Check that outliers were removed correctly
        self.assertEqual(df.shape, (4, 4))  # One row removed
        self.assertTrue(df['C'].max() < 100.0)  # Outlier value removed
    
    def test_encode_categorical(self):
        """Test encoding categorical variables."""
        self.processor.encode_categorical(columns=['B'])
        df = self.processor.get_data()
        
        # Check that categorical column was encoded correctly
        self.assertTrue('B_a' in df.columns)
        self.assertTrue('B_b' in df.columns)
        self.assertTrue('B_c' in df.columns)
        self.assertTrue('B_e' in df.columns)
        self.assertFalse('B' in df.columns)  # Original column removed
    
    def test_calculate_correlation(self):
        """Test calculating correlation."""
        corr = self.processor.calculate_correlation()
        
        # Check that correlation matrix was calculated correctly
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertEqual(corr.shape, (2, 2))  # Only numeric columns A and C
        self.assertTrue('A' in corr.columns)
        self.assertTrue('C' in corr.columns)
    
    def test_filter_data(self):
        """Test filtering data."""
        self.processor.filter_data('A > 2')
        df = self.processor.get_data()
        
        # Check that data was filtered correctly
        self.assertEqual(df.shape, (2, 4))  # Only rows where A > 2
        self.assertTrue(all(df['A'] > 2))
    
    def test_group_by(self):
        """Test grouping data."""
        grouped = self.processor.group_by('D', agg_dict={'A': 'mean', 'C': 'sum'})
        
        # Check that data was grouped correctly
        self.assertIsInstance(grouped, pd.DataFrame)
        self.assertEqual(grouped.shape, (2, 2))  # Two groups (True, False)
        self.assertTrue('A' in grouped.columns)
        self.assertTrue('C' in grouped.columns)

if __name__ == '__main__':
    unittest.main()

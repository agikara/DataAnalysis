"""
Data Analysis Tool - Tests for Visualizer Module

This module contains tests for the Visualizer class.
"""

import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock

# Import the module to test
from dataanalysistool.visualizer import Visualizer

class TestVisualizer(unittest.TestCase):
    """Test cases for Visualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = Visualizer()
        
        # Create test data
        self.test_df = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', periods=10),
            'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
            'metric': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        # Set data for visualizer
        self.visualizer.set_data(self.test_df)
    
    def tearDown(self):
        """Tear down test fixtures."""
        plt.close('all')  # Close all matplotlib figures
    
    def test_set_data(self):
        """Test setting data."""
        # Create new data
        new_df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6]
        })
        
        # Set new data
        self.visualizer.set_data(new_df)
        
        # Check that data was set correctly
        self.assertEqual(self.visualizer.data.shape, (3, 2))
        self.assertEqual(list(self.visualizer.data.columns), ['x', 'y'])
    
    def test_set_style(self):
        """Test setting style."""
        # Set style
        self.visualizer.set_style('darkgrid')
        
        # Not much to assert here, just make sure it doesn't raise an exception
        self.assertTrue(True)
    
    def test_line_plot_static(self):
        """Test creating a static line plot."""
        # Create plot
        fig = self.visualizer.line_plot(x='date', y='value', title='Test Line Plot', interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_line_plot_interactive(self):
        """Test creating an interactive line plot."""
        # Create plot
        fig = self.visualizer.line_plot(x='date', y='value', title='Test Line Plot', interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_bar_plot_static(self):
        """Test creating a static bar plot."""
        # Create plot
        fig = self.visualizer.bar_plot(x='category', y='value', title='Test Bar Plot', interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_bar_plot_interactive(self):
        """Test creating an interactive bar plot."""
        # Create plot
        fig = self.visualizer.bar_plot(x='category', y='value', title='Test Bar Plot', interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_scatter_plot_static(self):
        """Test creating a static scatter plot."""
        # Create plot
        fig = self.visualizer.scatter_plot(x='value', y='metric', title='Test Scatter Plot', interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_scatter_plot_interactive(self):
        """Test creating an interactive scatter plot."""
        # Create plot
        fig = self.visualizer.scatter_plot(x='value', y='metric', title='Test Scatter Plot', interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_histogram_static(self):
        """Test creating a static histogram."""
        # Create plot
        fig = self.visualizer.histogram(column='value', title='Test Histogram', interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_histogram_interactive(self):
        """Test creating an interactive histogram."""
        # Create plot
        fig = self.visualizer.histogram(column='value', title='Test Histogram', interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_box_plot_static(self):
        """Test creating a static box plot."""
        # Create plot
        fig = self.visualizer.box_plot(x='category', y='value', title='Test Box Plot', interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_box_plot_interactive(self):
        """Test creating an interactive box plot."""
        # Create plot
        fig = self.visualizer.box_plot(x='category', y='value', title='Test Box Plot', interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_heatmap_static(self):
        """Test creating a static heatmap."""
        # Create plot
        fig = self.visualizer.heatmap(title='Test Heatmap', interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_heatmap_interactive(self):
        """Test creating an interactive heatmap."""
        # Create plot
        fig = self.visualizer.heatmap(title='Test Heatmap', interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_pair_plot(self):
        """Test creating a pair plot."""
        # Create plot
        fig = self.visualizer.pair_plot(columns=['value', 'metric'], title='Test Pair Plot')
        
        # Check that a seaborn PairGrid was returned
        self.assertTrue(hasattr(fig, 'fig'))
    
    @patch('dataanalysistool.visualizer.plt.savefig')
    def test_save_plot_matplotlib(self, mock_savefig):
        """Test saving a matplotlib plot."""
        # Create plot
        fig = self.visualizer.line_plot(x='date', y='value', interactive=False)
        
        # Save plot
        self.visualizer.save_plot(fig, 'test_plot.png')
        
        # Check that savefig was called
        mock_savefig.assert_called_once()
    
    @patch('dataanalysistool.visualizer.go.Figure.write_image')
    def test_save_plot_plotly(self, mock_write_image):
        """Test saving a plotly plot."""
        # Create plot
        fig = self.visualizer.line_plot(x='date', y='value', interactive=True)
        
        # Save plot
        self.visualizer.save_plot(fig, 'test_plot.png')
        
        # Check that write_image was called
        mock_write_image.assert_called_once()
    
    def test_plot_distribution_static(self):
        """Test creating a static distribution plot."""
        # Create plot
        fig = self.visualizer.plot_distribution(column='value', title='Test Distribution', interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_distribution_interactive(self):
        """Test creating an interactive distribution plot."""
        # Create plot
        fig = self.visualizer.plot_distribution(column='value', title='Test Distribution', interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)
    
    def test_plot_time_series_static(self):
        """Test creating a static time series plot."""
        # Create plot
        fig = self.visualizer.plot_time_series(date_column='date', value_columns='value', 
                                              title='Test Time Series', interactive=False)
        
        # Check that a matplotlib figure was returned
        self.assertIsInstance(fig, plt.Figure)
    
    def test_plot_time_series_interactive(self):
        """Test creating an interactive time series plot."""
        # Create plot
        fig = self.visualizer.plot_time_series(date_column='date', value_columns='value', 
                                              title='Test Time Series', interactive=True)
        
        # Check that a plotly figure was returned
        self.assertIsInstance(fig, go.Figure)

if __name__ == '__main__':
    unittest.main()

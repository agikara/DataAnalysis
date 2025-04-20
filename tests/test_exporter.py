"""
Data Analysis Tool - Tests for Exporter Module

This module contains tests for the Exporter class.
"""

import os
import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock

# Import the module to test
from dataanalysistool.exporter import Exporter

class TestExporter(unittest.TestCase):
    """Test cases for Exporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test directory
        self.test_dir = os.path.join(os.path.dirname(__file__), 'test_output')
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create exporter with test directory
        self.exporter = Exporter(output_dir=self.test_dir)
        
        # Create test data
        self.test_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.1, 2.2, 3.3]
        })
        
        # Create test figure (matplotlib)
        self.fig_mpl = plt.figure()
        plt.plot([1, 2, 3], [4, 5, 6])
        
        # Create test figure (plotly)
        self.fig_plotly = go.Figure()
        self.fig_plotly.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    
    def tearDown(self):
        """Tear down test fixtures."""
        plt.close('all')  # Close all matplotlib figures
        
        # Remove test files
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        
        # Remove test directory
        os.rmdir(self.test_dir)
    
    def test_export_csv(self):
        """Test exporting to CSV."""
        # Export to CSV
        filepath = self.exporter.export_csv(self.test_df, 'test_export')
        
        # Check that file was created
        self.assertTrue(os.path.exists(filepath))
        
        # Check that file has correct content
        df_loaded = pd.read_csv(filepath)
        self.assertEqual(df_loaded.shape, (3, 3))
        self.assertEqual(list(df_loaded.columns), ['A', 'B', 'C'])
    
    def test_export_excel(self):
        """Test exporting to Excel."""
        # Create data dictionary
        data_dict = {
            'Sheet1': self.test_df,
            'Sheet2': self.test_df.copy()
        }
        
        # Export to Excel
        filepath = self.exporter.export_excel(data_dict, 'test_export')
        
        # Check that file was created
        self.assertTrue(os.path.exists(filepath))
        
        # Check that file has correct content
        df_loaded = pd.read_excel(filepath, sheet_name='Sheet1')
        self.assertEqual(df_loaded.shape, (3, 3))
        self.assertEqual(list(df_loaded.columns), ['A', 'B', 'C'])
    
    def test_export_json(self):
        """Test exporting to JSON."""
        # Export to JSON
        filepath = self.exporter.export_json(self.test_df, 'test_export')
        
        # Check that file was created
        self.assertTrue(os.path.exists(filepath))
        
        # Check that file has correct content
        df_loaded = pd.read_json(filepath)
        self.assertEqual(df_loaded.shape, (3, 3))
        self.assertEqual(sorted(list(df_loaded.columns)), ['A', 'B', 'C'])
    
    @patch('dataanalysistool.exporter.plt.Figure.savefig')
    def test_export_plot_matplotlib(self, mock_savefig):
        """Test exporting a matplotlib plot."""
        # Export plot
        filepath = self.exporter.export_plot(self.fig_mpl, 'test_plot', format='png')
        
        # Check that savefig was called
        mock_savefig.assert_called_once()
    
    @patch('dataanalysistool.exporter.go.Figure.write_image')
    def test_export_plot_plotly(self, mock_write_image):
        """Test exporting a plotly plot."""
        # Export plot
        filepath = self.exporter.export_plot(self.fig_plotly, 'test_plot', format='png')
        
        # Check that write_image was called
        mock_write_image.assert_called_once()
    
    @patch('dataanalysistool.exporter.Exporter._figure_to_base64')
    def test_export_html_report(self, mock_to_base64):
        """Test exporting an HTML report."""
        # Mock the base64 conversion
        mock_to_base64.return_value = 'data:image/png;base64,abc123'
        
        # Create content blocks
        content_blocks = [
            {'type': 'heading', 'content': 'Test Report'},
            {'type': 'text', 'content': 'This is a test report.'},
            {'type': 'figure', 'content': self.fig_mpl, 'caption': 'Test Figure'},
            {'type': 'dataframe', 'content': self.test_df, 'caption': 'Test Data'}
        ]
        
        # Export report
        filepath = self.exporter.export_html_report('Test Report', content_blocks, 'test_report')
        
        # Check that file was created
        self.assertTrue(os.path.exists(filepath))
        
        # Check that file has correct content
        with open(filepath, 'r') as f:
            content = f.read()
        
        self.assertIn('Test Report', content)
        self.assertIn('This is a test report.', content)
        self.assertIn('Test Figure', content)
        self.assertIn('Test Data', content)
    
    @patch('dataanalysistool.exporter.pdfkit.from_file')
    @patch('dataanalysistool.exporter.Exporter.export_html_report')
    def test_export_pdf_report(self, mock_export_html, mock_from_file):
        """Test exporting a PDF report."""
        # Mock the HTML export
        mock_export_html.return_value = os.path.join(self.test_dir, 'test_report.html')
        
        # Create content blocks
        content_blocks = [
            {'type': 'heading', 'content': 'Test Report'},
            {'type': 'text', 'content': 'This is a test report.'}
        ]
        
        # Export report
        filepath = self.exporter.export_pdf_report('Test Report', content_blocks, 'test_report')
        
        # Check that export_html_report was called
        mock_export_html.assert_called_once()
        
        # Check that from_file was called
        mock_from_file.assert_called_once()
    
    @patch('dataanalysistool.exporter.Exporter.export_html_report')
    def test_export_analysis_results_html(self, mock_export_html):
        """Test exporting analysis results to HTML."""
        # Create mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.results = {
            'performance_metrics': {
                'ticker': 'TEST',
                'period': '2020-01-01 to 2020-04-09 (100 days)',
                'cumulative_return': 0.5,
                'annualized_return': 0.2,
                'annualized_volatility': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.1,
                'max_drawdown_duration': 10
            }
        }
        
        # Mock the plot methods
        mock_analyzer.plot_price_history.return_value = self.fig_plotly
        mock_analyzer.plot_returns_distribution.return_value = self.fig_plotly
        mock_analyzer.plot_cumulative_returns.return_value = self.fig_plotly
        mock_analyzer.plot_drawdown.return_value = self.fig_plotly
        
        # Export results
        self.exporter.export_analysis_results(mock_analyzer, 'test_analysis', format='html')
        
        # Check that export_html_report was called
        mock_export_html.assert_called_once()
    
    def test_export_analysis_results_json(self):
        """Test exporting analysis results to JSON."""
        # Create mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.results = {
            'performance_metrics': {
                'ticker': 'TEST',
                'period': '2020-01-01 to 2020-04-09 (100 days)',
                'cumulative_return': 0.5,
                'annualized_return': 0.2,
                'annualized_volatility': 0.15,
                'sharpe_ratio': 1.2,
                'max_drawdown': -0.1,
                'max_drawdown_duration': 10
            },
            'daily_returns': pd.Series([0.01, 0.02, -0.01])
        }
        
        # Export results
        filepath = self.exporter.export_analysis_results(mock_analyzer, 'test_analysis', format='json')
        
        # Check that file was created
        self.assertTrue(os.path.exists(filepath))
        
        # Check that file has correct content
        with open(filepath, 'r') as f:
            import json
            content = json.load(f)
        
        self.assertTrue('performance_metrics' in content)
        self.assertTrue('daily_returns' in content)
        self.assertEqual(content['performance_metrics']['ticker'], 'TEST')

if __name__ == '__main__':
    unittest.main()

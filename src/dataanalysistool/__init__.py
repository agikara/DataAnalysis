"""
Data Analysis Tool - Main Package Initialization

This module initializes the Data Analysis Tool package.
"""

__version__ = '0.1.0'
__author__ = 'Data Analysis Tool Team'
__email__ = 'support@dataanalysistool.com'
__description__ = 'A comprehensive Python-based data analysis tool'

# Import main components for easier access
from .data_loader import DataLoader
from .data_processor import DataProcessor
from .visualizer import Visualizer
from .financial import FinancialAnalyzer
from .exporter import Exporter
from .utils import setup_logger

# Set up default logger
setup_logger()

# Define package exports
__all__ = [
    'DataLoader',
    'DataProcessor',
    'Visualizer',
    'FinancialAnalyzer',
    'Exporter',
    'setup_logger'
]

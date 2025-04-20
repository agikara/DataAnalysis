"""
Data Analysis Tool - Configuration Module

This module contains configuration settings for the Data Analysis Tool.
"""

# Default configuration settings
DEFAULT_CONFIG = {
    # General settings
    "default_data_dir": "data",
    "default_output_dir": "output",
    "log_level": "INFO",
    
    # Data loading settings
    "csv_encoding": "utf-8",
    "excel_engine": "openpyxl",
    "date_format": "%Y-%m-%d",
    
    # Visualization settings
    "default_figsize": (10, 6),
    "default_dpi": 100,
    "default_theme": "whitegrid",
    "color_palette": "viridis",
    
    # Financial analysis settings
    "default_ticker": "AAPL",
    "default_period": "1y",
    "default_interval": "1d",
    
    # Web dashboard settings
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True,
}

# User configuration (can be overridden)
user_config = {}

def get_config(key=None):
    """
    Get configuration value.
    
    Args:
        key (str, optional): Configuration key. If None, returns entire config.
        
    Returns:
        The configuration value or the entire config dictionary.
    """
    config = {**DEFAULT_CONFIG, **user_config}
    if key is None:
        return config
    return config.get(key)

def set_config(key, value):
    """
    Set configuration value.
    
    Args:
        key (str): Configuration key.
        value: Configuration value.
    """
    user_config[key] = value

def reset_config():
    """Reset user configuration to defaults."""
    user_config.clear()

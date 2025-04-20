# Data Analysis Tool - Setup Documentation

## Overview

The Data Analysis Tool is a comprehensive Python-based application designed for data analysis, visualization, and financial analysis. It provides a robust set of features for loading, processing, visualizing, and analyzing data from various sources, with special capabilities for financial data analysis.

## System Requirements

- Python 3.8 or higher
- Operating System: Windows, macOS, or Linux
- Minimum 4GB RAM (8GB recommended for larger datasets)
- 500MB free disk space

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install dataanalysistool
```

### Option 2: Install from Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DataAnalysisTool.git
cd DataAnalysisTool
```

2. Install the package in development mode:
```bash
pip install -e .
```

### Dependencies

The following dependencies will be automatically installed:

- numpy
- pandas
- matplotlib
- seaborn
- plotly
- scikit-learn
- yfinance
- flask
- jupyter
- pytest
- sphinx
- pdfkit

## Quick Start

### Command Line Interface

The Data Analysis Tool provides a command-line interface for common operations:

1. **Load data**:
```bash
dataanalysistool load --file data.csv --type csv
```

2. **Process data**:
```bash
dataanalysistool process --file data.csv --normalize --missing mean
```

3. **Visualize data**:
```bash
dataanalysistool visualize --file data.csv --plot line --x date --y value --output plot.png
```

4. **Financial analysis**:
```bash
dataanalysistool finance --ticker AAPL --period 1y --output apple_analysis.html
```

5. **Generate report**:
```bash
dataanalysistool report --file data.csv --output report.pdf --format pdf
```

6. **Start web dashboard**:
```bash
dataanalysistool dashboard
```

### Python API

You can also use the Data Analysis Tool as a Python library:

```python
from dataanalysistool.data_loader import DataLoader
from dataanalysistool.data_processor import DataProcessor
from dataanalysistool.visualizer import Visualizer
from dataanalysistool.financial import FinancialAnalyzer
from dataanalysistool.exporter import Exporter

# Load data
loader = DataLoader()
data = loader.load_csv('data.csv')

# Process data
processor = DataProcessor(data)
processor.handle_missing_values(strategy='mean')
processor.normalize()
processed_data = processor.get_data()

# Visualize data
visualizer = Visualizer(processed_data)
fig = visualizer.line_plot(x='date', y='value', title='My Plot', interactive=True)
visualizer.save_plot(fig, 'my_plot.png')

# Financial analysis
financial = FinancialAnalyzer()
stock_data = loader.load_stock_data('AAPL', period='1y')
financial.set_data(stock_data, ticker='AAPL')
metrics = financial.calculate_performance_metrics()
print(metrics)

# Export results
exporter = Exporter(output_dir='output')
exporter.export_analysis_results(financial, 'apple_analysis', format='html')
```

### Web Dashboard

The Data Analysis Tool includes a web dashboard for interactive data analysis:

1. Start the dashboard:
```bash
dataanalysistool dashboard
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Use the web interface to:
   - Upload data files
   - Process and clean data
   - Create visualizations
   - Perform financial analysis
   - Generate reports

## Directory Structure

```
DataAnalysisTool/
├── docs/                  # Documentation
├── src/                   # Source code
│   └── dataanalysistool/  # Main package
│       ├── __init__.py    # Package initialization
│       ├── config.py      # Configuration module
│       ├── data_loader.py # Data loading module
│       ├── data_processor.py # Data processing module
│       ├── visualizer.py  # Visualization module
│       ├── financial.py   # Financial analysis module
│       ├── exporter.py    # Export module
│       ├── utils.py       # Utility functions
│       └── ui/            # User interfaces
│           ├── cli.py     # Command-line interface
│           └── web.py     # Web dashboard
├── tests/                 # Unit tests
├── data/                  # Sample data
├── README.md              # Project readme
├── setup.py               # Package setup script
└── requirements.txt       # Dependencies
```

## Configuration

The Data Analysis Tool can be configured by creating a configuration file at `~/.dataanalysistool/config.json`:

```json
{
  "default_output_dir": "/path/to/output",
  "default_theme": "darkgrid",
  "default_figsize": [10, 6],
  "default_dpi": 100,
  "color_palette": "deep",
  "log_level": "INFO"
}
```

## Advanced Usage

### Custom Data Loaders

You can extend the `DataLoader` class to support additional data sources:

```python
from dataanalysistool.data_loader import DataLoader

class MyCustomLoader(DataLoader):
    def load_custom_format(self, filepath):
        # Custom loading logic
        data = ...
        return data
```

### Custom Visualizations

You can extend the `Visualizer` class to create custom visualizations:

```python
from dataanalysistool.visualizer import Visualizer

class MyCustomVisualizer(Visualizer):
    def custom_plot(self, x, y, **kwargs):
        # Custom plotting logic
        fig = ...
        return fig
```

### Batch Processing

For batch processing of multiple files:

```python
import os
from dataanalysistool.data_loader import DataLoader
from dataanalysistool.data_processor import DataProcessor
from dataanalysistool.exporter import Exporter

loader = DataLoader()
processor = DataProcessor()
exporter = Exporter(output_dir='processed')

for filename in os.listdir('data'):
    if filename.endswith('.csv'):
        # Load data
        data = loader.load_csv(os.path.join('data', filename))
        
        # Process data
        processor.set_data(data)
        processor.handle_missing_values()
        processor.normalize()
        processed_data = processor.get_data()
        
        # Export processed data
        output_filename = f"processed_{filename}"
        exporter.export_csv(processed_data, output_filename)
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'dataanalysistool'**
   - Ensure the package is installed correctly
   - Check your Python path

2. **FileNotFoundError when loading data**
   - Verify the file path is correct
   - Use absolute paths if necessary

3. **Web dashboard not starting**
   - Check if port 5000 is already in use
   - Try specifying a different port: `dataanalysistool dashboard --port 5001`

4. **PDF export not working**
   - Ensure wkhtmltopdf is installed on your system
   - On Ubuntu: `sudo apt-get install wkhtmltopdf`
   - On macOS: `brew install wkhtmltopdf`
   - On Windows: Download from https://wkhtmltopdf.org/downloads.html

### Getting Help

If you encounter issues not covered in this documentation:

1. Check the [GitHub repository](https://github.com/yourusername/DataAnalysisTool) for known issues
2. Submit a bug report or feature request
3. Contact the maintainers at support@dataanalysistool.com

## License

This project is licensed under the MIT License - see the LICENSE file for details.

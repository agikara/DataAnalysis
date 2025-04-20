"""
Data Analysis Tool - Exporter Module

This module provides functionality for exporting analysis results.
"""
import pdfkit

# Configure pdfkit to use wkhtmltopdf
pdfkit_config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')  # Update this path!
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
from datetime import datetime
import json
import base64
from io import BytesIO
import jinja2
import pdfkit
from .config import get_config

class Exporter:
    """
    Class for exporting analysis results.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the Exporter.
        
        Args:
            output_dir (str, optional): Directory to save exported files.
        """
        if output_dir is None:
            output_dir = get_config('default_output_dir')
        
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_csv(self, data, filename, index=True):
        """
        Export data to CSV.
        
        Args:
            data (pandas.DataFrame): Data to export.
            filename (str): Filename to save to.
            index (bool, optional): Whether to include index in the output.
            
        Returns:
            str: Path to the exported file.
        """
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        filepath = os.path.join(self.output_dir, filename)
        data.to_csv(filepath, index=index)
        
        return filepath
    
    def export_excel(self, data_dict, filename, index=True):
        """
        Export data to Excel.
        
        Args:
            data_dict (dict): Dictionary mapping sheet names to DataFrames.
            filename (str): Filename to save to.
            index (bool, optional): Whether to include index in the output.
            
        Returns:
            str: Path to the exported file.
        """
        if not filename.endswith('.xlsx'):
            filename += '.xlsx'
        
        filepath = os.path.join(self.output_dir, filename)
        
        with pd.ExcelWriter(filepath) as writer:
            for sheet_name, data in data_dict.items():
                data.to_excel(writer, sheet_name=sheet_name, index=index)
        
        return filepath
    
    def export_json(self, data, filename, orient='records'):
        """
        Export data to JSON.
        
        Args:
            data (pandas.DataFrame or dict): Data to export.
            filename (str): Filename to save to.
            orient (str, optional): Format of JSON data if data is a DataFrame.
            
        Returns:
            str: Path to the exported file.
        """
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = os.path.join(self.output_dir, filename)
        
        if isinstance(data, pd.DataFrame):
            data.to_json(filepath, orient=orient)
        else:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=4)
        
        return filepath
    
    def export_plot(self, plot, filename, format='png', dpi=None):
        """
        Export a plot to a file.
        
        Args:
            plot: Plot object to save.
            filename (str): Filename to save to.
            format (str, optional): File format.
            dpi (int, optional): DPI for raster formats.
            
        Returns:
            str: Path to the exported file.
        """
        if not filename.endswith(f'.{format}'):
            filename += f'.{format}'
        
        filepath = os.path.join(self.output_dir, filename)
        
        if dpi is None:
            dpi = get_config('default_dpi')
        
        if hasattr(plot, 'write_image'):
            # Plotly figure
            plot.write_image(filepath, format=format)
        elif hasattr(plot, 'savefig'):
            # Matplotlib figure
            plot.savefig(filepath, format=format, dpi=dpi)
        elif hasattr(plot, 'fig'):
            # Seaborn grid
            plot.fig.savefig(filepath, format=format, dpi=dpi)
        else:
            raise ValueError("Unsupported plot type")
        
        return filepath
    
    def _figure_to_base64(self, fig, format='png', dpi=None):
        """
        Convert a figure to base64 encoded string.
        
        Args:
            fig: Figure to convert.
            format (str, optional): Image format.
            dpi (int, optional): DPI for raster formats.
            
        Returns:
            str: Base64 encoded image.
        """
        if dpi is None:
            dpi = get_config('default_dpi')
        
        buf = BytesIO()
        
        if hasattr(fig, 'write_image'):
            # Plotly figure
            fig.write_image(buf, format=format)
        elif hasattr(fig, 'savefig'):
            # Matplotlib figure
            fig.savefig(buf, format=format, dpi=dpi)
        elif hasattr(fig, 'fig'):
            # Seaborn grid
            fig.fig.savefig(buf, format=format, dpi=dpi)
        else:
            raise ValueError("Unsupported figure type")
        
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        
        return f"data:image/{format};base64,{img_data}"
    
    def export_html_report(self, title, content_blocks, filename, template=None):
        """
        Export an HTML report.
        
        Args:
            title (str): Report title.
            content_blocks (list): List of content blocks (dicts with type and content).
            filename (str): Filename to save to.
            template (str, optional): Custom Jinja2 template.
            
        Returns:
            str: Path to the exported file.
        """
        if not filename.endswith('.html'):
            filename += '.html'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Process content blocks
        processed_blocks = []
        for block in content_blocks:
            block_type = block.get('type', 'text')
            content = block.get('content')
            
            if block_type == 'figure' and content is not None:
                # Convert figure to base64
                img_data = self._figure_to_base64(content)
                processed_blocks.append({
                    'type': 'image',
                    'content': img_data,
                    'caption': block.get('caption', '')
                })
            elif block_type == 'dataframe' and content is not None:
                # Convert DataFrame to HTML
                df_html = content.to_html(classes='dataframe')
                processed_blocks.append({
                    'type': 'html',
                    'content': df_html,
                    'caption': block.get('caption', '')
                })
            else:
                # Pass through other block types
                processed_blocks.append(block)
        
        # Default template if none provided
        if template is None:
            template = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>{{ title }}</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        margin: 0;
                        padding: 20px;
                        color: #333;
                    }
                    .container {
                        max-width: 1200px;
                        margin: 0 auto;
                    }
                    h1, h2, h3, h4 {
                        color: #2c3e50;
                    }
                    .header {
                        text-align: center;
                        margin-bottom: 30px;
                        padding-bottom: 20px;
                        border-bottom: 1px solid #eee;
                    }
                    .content-block {
                        margin-bottom: 30px;
                    }
                    .caption {
                        font-style: italic;
                        color: #666;
                        text-align: center;
                        margin-top: 5px;
                    }
                    img {
                        max-width: 100%;
                        height: auto;
                        display: block;
                        margin: 0 auto;
                    }
                    table.dataframe {
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }
                    table.dataframe th, table.dataframe td {
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: left;
                    }
                    table.dataframe th {
                        background-color: #f2f2f2;
                    }
                    table.dataframe tr:nth-child(even) {
                        background-color: #f9f9f9;
                    }
                    .footer {
                        margin-top: 50px;
                        padding-top: 20px;
                        border-top: 1px solid #eee;
                        text-align: center;
                        font-size: 0.9em;
                        color: #777;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>{{ title }}</h1>
                        <p>Generated on {{ date }}</p>
                    </div>
                    
                    {% for block in blocks %}
                        <div class="content-block">
                            {% if block.type == 'heading' %}
                                <h{{ block.level|default(2) }}>{{ block.content }}</h{{ block.level|default(2) }}>
                            {% elif block.type == 'text' %}
                                <p>{{ block.content }}</p>
                            {% elif block.type == 'image' %}
                                <img src="{{ block.content }}" alt="{{ block.caption }}">
                                {% if block.caption %}
                                    <div class="caption">{{ block.caption }}</div>
                                {% endif %}
                            {% elif block.type == 'html' %}
                                {{ block.content|safe }}
                                {% if block.caption %}
                                    <div class="caption">{{ block.caption }}</div>
                                {% endif %}
                            {% endif %}
                        </div>
                    {% endfor %}
                    
                    <div class="footer">
                        <p>Report generated by Data Analysis Tool</p>
                    </div>
                </div>
            </body>
            </html>
            """
        
        # Render template
        env = jinja2.Environment()
        template = env.from_string(template)
        html = template.render(
            title=title,
            blocks=processed_blocks,
            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return filepath
    
    def export_pdf_report(self, title, content_blocks, filename, template=None, options=None):
        """
        Export a PDF report.
        
        Args:
            title (str): Report title.
            content_blocks (list): List of content blocks (dicts with type and content).
            filename (str): Filename to save to.
            template (str, optional): Custom Jinja2 template.
            options (dict, optional): Options for wkhtmltopdf.
            
        Returns:
            str: Path to the exported file.
        """
        if not filename.endswith('.pdf'):
            filename += '.pdf'
        
        filepath = os.path.join(self.output_dir, filename)
        
        # First export as HTML
        html_filename = os.path.splitext(filename)[0] + '.html'
        html_path = self.export_html_report(title, content_blocks, html_filename, template)
        
        # Default options for wkhtmltopdf
        if options is None:
            options = {
                'page-size': 'A4',
                'margin-top': '20mm',
                'margin-right': '20mm',
                'margin-bottom': '20mm',
                'margin-left': '20mm',
                'encoding': 'UTF-8',
                'no-outline': None,
                'enable-local-file-access': None
            }
        
        # Convert HTML to PDF
        pdfkit.from_file(html_path, filepath, options=options)
        
        return filepath
    
    def export_analysis_results(self, analyzer, filename_prefix, include_plots=True, format='html'):
        """
        Export analysis results from a data analyzer.
        
        Args:
            analyzer: Analyzer object with results attribute.
            filename_prefix (str): Prefix for output filenames.
            include_plots (bool, optional): Whether to include plots in the report.
            format (str, optional): Output format ('html', 'pdf', 'excel', 'json').
            
        Returns:
            str: Path to the exported file.
        """
        if not hasattr(analyzer, 'results'):
            raise ValueError("Analyzer must have a results attribute")
        
        # Extract results
        results = analyzer.results
        
        if format == 'json':
            # Convert results to JSON-serializable format
            json_results = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                    json_results[key] = value.to_dict()
                elif isinstance(value, dict):
                    json_results[key] = value
                elif isinstance(value, np.ndarray):
                    json_results[key] = value.tolist()
                elif hasattr(value, '__dict__'):
                    json_results[key] = str(value)
                else:
                    json_results[key] = value
            
            return self.export_json(json_results, f"{filename_prefix}_results")
        
        elif format == 'excel':
            # Export DataFrames to Excel
            excel_data = {}
            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    excel_data[key] = value
                elif isinstance(value, pd.Series):
                    excel_data[key] = pd.DataFrame(value)
                elif isinstance(value, dict) and all(isinstance(v, (pd.DataFrame, pd.Series)) for v in value.values()):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, pd.DataFrame):
                            excel_data[f"{key}_{subkey}"] = subvalue
                        elif isinstance(subvalue, pd.Series):
                            excel_data[f"{key}_{subkey}"] = pd.DataFrame(subvalue)
            
            return self.export_excel(excel_data, f"{filename_prefix}_results")
        
        elif format in ['html', 'pdf']:
            # Create content blocks for report
            content_blocks = [
                {'type': 'heading', 'content': 'Analysis Results', 'level': 1}
            ]
            
            # Add summary if available
            if hasattr(analyzer, 'get_data_summary'):
                summary = analyzer.get_data_summary()
                content_blocks.append({'type': 'heading', 'content': 'Data Summary', 'level': 2})
                content_blocks.append({'type': 'text', 'content': f"Source: {summary.get('source', 'Unknown')}"})
                content_blocks.append({'type': 'text', 'content': f"Shape: {summary.get('shape', 'Unknown')}"})
                content_blocks.append({'type': 'text', 'content': f"Columns: {', '.join(summary.get('columns', []))}"})
            
            # Add performance metrics if available
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                content_blocks.append({'type': 'heading', 'content': 'Performance Metrics', 'level': 2})
                
                # Create a DataFrame for metrics
                metrics_df = pd.DataFrame({
                    'Metric': [
                        'Ticker',
                        'Period',
                        'Cumulative Return',
                        'Annualized Return',
                        'Annualized Volatility',
                        'Sharpe Ratio',
                        'Maximum Drawdown',
                        'Maximum Drawdown Duration'
                    ],
                    'Value': [
                        metrics.get('ticker', 'Unknown'),
                        metrics.get('period', 'Unknown'),
                        metrics.get('cumulative_return', 'Unknown'),
                        metrics.get('annualized_return', 'Unknown'),
                        metrics.get('annualized_volatility', 'Unknown'),
                        metrics.get('sharpe_ratio', 'Unknown'),
                        metrics.get('max_drawdown', 'Unknown'),
                        f"{metrics.get('max_drawdown_duration', 'Unknown')} days"
                    ]
                })
                
                content_blocks.append({'type': 'dataframe', 'content': metrics_df})
            
            # Add plots if requested
            if include_plots and hasattr(analyzer, 'plot_price_history'):
                content_blocks.append({'type': 'heading', 'content': 'Price History', 'level': 2})
                fig = analyzer.plot_price_history(interactive=True)
                content_blocks.append({'type': 'figure', 'content': fig, 'caption': 'Price History'})
            
            if include_plots and hasattr(analyzer, 'plot_returns_distribution'):
                content_blocks.append({'type': 'heading', 'content': 'Returns Distribution', 'level': 2})
                fig = analyzer.plot_returns_distribution(interactive=True)
                content_blocks.append({'type': 'figure', 'content': fig, 'caption': 'Returns Distribution'})
            
            if include_plots and hasattr(analyzer, 'plot_cumulative_returns'):
                content_blocks.append({'type': 'heading', 'content': 'Cumulative Returns', 'level': 2})
                fig = analyzer.plot_cumulative_returns(interactive=True)
                content_blocks.append({'type': 'figure', 'content': fig, 'caption': 'Cumulative Returns'})
            
            if include_plots and hasattr(analyzer, 'plot_drawdown'):
                content_blocks.append({'type': 'heading', 'content': 'Drawdown', 'level': 2})
                fig = analyzer.plot_drawdown(interactive=True)
                content_blocks.append({'type': 'figure', 'content': fig, 'caption': 'Drawdown'})
            
            if include_plots and hasattr(analyzer, 'plot_technical_indicators'):
                content_blocks.append({'type': 'heading', 'content': 'Technical Indicators', 'level': 2})
                fig = analyzer.plot_technical_indicators(interactive=True)
                content_blocks.append({'type': 'figure', 'content': fig, 'caption': 'Technical Indicators'})
            
            # Export report
            if format == 'html':
                return self.export_html_report('Analysis Results', content_blocks, f"{filename_prefix}_report")
            else:  # pdf
                return self.export_pdf_report('Analysis Results', content_blocks, f"{filename_prefix}_report")
        
        else:
            raise ValueError(f"Unsupported format: {format}")

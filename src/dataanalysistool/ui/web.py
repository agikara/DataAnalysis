"""
Data Analysis Tool - UI Module - Web Dashboard

This module provides a web-based dashboard for the Data Analysis Tool.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from .. import config
from ..data_loader import DataLoader
from ..data_processor import DataProcessor
from ..visualizer import Visualizer
from ..financial import FinancialAnalyzer
from ..exporter import Exporter
from ..utils import setup_logger, create_sample_data, create_sample_stock_data

logger = logging.getLogger('dataanalysistool')

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: The configured Flask application.
    """
    # Create Flask app
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
                static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    
    # Configure app
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
    app.config['SECRET_KEY'] = os.urandom(24)
    
    # Create upload folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Create instances of tool components
    loader = DataLoader()
    processor = DataProcessor()
    visualizer = Visualizer()
    financial = FinancialAnalyzer()
    exporter = Exporter()
    
    # Set up logging
    setup_logger()
    
    # Define allowed file extensions
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json'}
    
    def allowed_file(filename):
        """Check if a file has an allowed extension."""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    @app.route('/')
    def index():
        """Render the dashboard home page."""
        return render_template('index.html')
    
    @app.route('/upload', methods=['GET', 'POST'])
    def upload_file():
        """Handle file uploads."""
        if request.method == 'POST':
            # Check if the post request has the file part
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            
            file = request.files['file']
            
            # If user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Load the data
                try:
                    if filename.endswith('.csv'):
                        data = loader.load_csv(filepath)
                    elif filename.endswith('.xlsx'):
                        data = loader.load_excel(filepath)
                    elif filename.endswith('.json'):
                        data = loader.load_json(filepath)
                    else:
                        return jsonify({'error': 'Unsupported file type'}), 400
                    
                    # Store data summary
                    summary = {
                        'filename': filename,
                        'filepath': filepath,
                        'shape': data.shape,
                        'columns': list(data.columns),
                        'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                        'head': data.head().to_html(classes='table table-striped')
                    }
                    
                    return jsonify({
                        'success': True,
                        'filename': filename,
                        'summary': summary
                    })
                
                except Exception as e:
                    logger.error(f"Error loading data: {e}")
                    return jsonify({'error': f'Error loading data: {str(e)}'}), 500
            
            return jsonify({'error': 'File type not allowed'}), 400
        
        return render_template('upload.html')
    
    @app.route('/data/<filename>')
    def data_view(filename):
        """View data details."""
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(filepath):
            return render_template('error.html', message=f"File not found: {filename}")
        
        try:
            # Load the data
            if filename.endswith('.csv'):
                data = loader.load_csv(filepath)
            elif filename.endswith('.xlsx'):
                data = loader.load_excel(filepath)
            elif filename.endswith('.json'):
                data = loader.load_json(filepath)
            else:
                return render_template('error.html', message=f"Unsupported file type: {filename}")
            
            # Get data summary
            summary = {
                'filename': filename,
                'filepath': filepath,
                'shape': data.shape,
                'columns': list(data.columns),
                'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
                'missing': data.isna().sum().to_dict(),
                'head': data.head(10).to_html(classes='table table-striped'),
                'describe': data.describe().to_html(classes='table table-striped')
            }
            
            return render_template('data_view.html', summary=summary)
        
        except Exception as e:
            logger.error(f"Error viewing data: {e}")
            return render_template('error.html', message=f"Error viewing data: {str(e)}")
    
    @app.route('/process/<filename>', methods=['GET', 'POST'])
    def process_data(filename):
        """Process data."""
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(filepath):
            return render_template('error.html', message=f"File not found: {filename}")
        
        try:
            # Load the data
            if filename.endswith('.csv'):
                data = loader.load_csv(filepath)
            elif filename.endswith('.xlsx'):
                data = loader.load_excel(filepath)
            elif filename.endswith('.json'):
                data = loader.load_json(filepath)
            else:
                return render_template('error.html', message=f"Unsupported file type: {filename}")
            
            # Set data for processor
            processor.set_data(data)
            
            if request.method == 'POST':
                # Process data based on form inputs
                if 'missing_strategy' in request.form and request.form['missing_strategy'] != 'none':
                    strategy = request.form['missing_strategy']
                    if strategy == 'drop':
                        data = data.dropna()
                        processor.set_data(data)
                    else:
                        processor.handle_missing_values(strategy=strategy)
                
                if 'normalize' in request.form and request.form['normalize'] == 'yes':
                    processor.normalize()
                
                if 'remove_outliers' in request.form and request.form['remove_outliers'] == 'yes':
                    processor.remove_outliers()
                
                # Get processed data
                processed_data = processor.get_data()
                
                # Save processed data
                processed_filename = f"processed_{filename}"
                processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
                
                if processed_filename.endswith('.csv'):
                    processed_data.to_csv(processed_filepath, index=False)
                elif processed_filename.endswith('.xlsx'):
                    processed_data.to_excel(processed_filepath, index=False)
                elif processed_filename.endswith('.json'):
                    processed_data.to_json(processed_filepath, orient='records')
                
                return redirect(url_for('data_view', filename=processed_filename))
            
            # Get column information for the form
            columns = list(data.columns)
            numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
            
            return render_template('process.html', filename=filename, columns=columns, 
                                  numeric_columns=numeric_columns)
        
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return render_template('error.html', message=f"Error processing data: {str(e)}")
    
    @app.route('/visualize/<filename>', methods=['GET', 'POST'])
    def visualize_data(filename):
        """Create visualizations."""
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(filepath):
            return render_template('error.html', message=f"File not found: {filename}")
        
        try:
            # Load the data
            if filename.endswith('.csv'):
                data = loader.load_csv(filepath)
            elif filename.endswith('.xlsx'):
                data = loader.load_excel(filepath)
            elif filename.endswith('.json'):
                data = loader.load_json(filepath)
            else:
                return render_template('error.html', message=f"Unsupported file type: {filename}")
            
            # Set data for visualizer
            visualizer.set_data(data)
            
            if request.method == 'POST':
                # Create visualization based on form inputs
                plot_type = request.form['plot_type']
                x_column = request.form.get('x_column')
                y_column = request.form.get('y_column')
                title = request.form.get('title', f"{plot_type.capitalize()} Plot")
                
                # Create the plot
                if plot_type == 'line':
                    fig = visualizer.line_plot(x=x_column, y=y_column, title=title, interactive=True)
                elif plot_type == 'bar':
                    fig = visualizer.bar_plot(x=x_column, y=y_column, title=title, interactive=True)
                elif plot_type == 'scatter':
                    fig = visualizer.scatter_plot(x=x_column, y=y_column, title=title, interactive=True)
                elif plot_type == 'histogram':
                    fig = visualizer.histogram(column=x_column, title=title, interactive=True)
                elif plot_type == 'box':
                    fig = visualizer.box_plot(x=x_column, y=y_column, title=title, interactive=True)
                elif plot_type == 'heatmap':
                    fig = visualizer.heatmap(title=title, interactive=True)
                else:
                    return render_template('error.html', message=f"Unsupported plot type: {plot_type}")
                
                # Convert plot to JSON for rendering
                plot_json = fig.to_json()
                
                return render_template('visualization.html', filename=filename, plot_json=plot_json,
                                      plot_type=plot_type, title=title)
            
            # Get column information for the form
            columns = list(data.columns)
            numeric_columns = list(data.select_dtypes(include=[np.number]).columns)
            
            return render_template('visualize.html', filename=filename, columns=columns, 
                                  numeric_columns=numeric_columns)
        
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return render_template('error.html', message=f"Error creating visualization: {str(e)}")
    
    @app.route('/finance', methods=['GET', 'POST'])
    def finance():
        """Perform financial analysis."""
        if request.method == 'POST':
            try:
                # Get form inputs
                ticker = request.form['ticker']
                period = request.form.get('period', '1y')
                interval = request.form.get('interval', '1d')
                
                # Load stock data
                data = loader.load_stock_data(ticker, period=period, interval=interval)
                
                # Set data for financial analyzer
                financial.set_data(data, ticker=ticker)
                
                # Calculate performance metrics
                metrics = financial.calculate_performance_metrics()
                
                # Create plots
                price_plot = financial.plot_price_history(interactive=True)
                returns_plot = financial.plot_returns_distribution(interactive=True)
                cumulative_plot = financial.plot_cumulative_returns(interactive=True)
                drawdown_plot = financial.plot_drawdown(interactive=True)
                
                # Convert plots to JSON for rendering
                plots = {
                    'price': price_plot.to_json(),
                    'returns': returns_plot.to_json(),
                    'cumulative': cumulative_plot.to_json(),
                    'drawdown': drawdown_plot.to_json()
                }
                
                return render_template('finance_results.html', ticker=ticker, metrics=metrics, plots=plots)
            
            except Exception as e:
                logger.error(f"Error performing financial analysis: {e}")
                return render_template('error.html', message=f"Error performing financial analysis: {str(e)}")
        
        return render_template('finance.html')
    
    @app.route('/report/<filename>', methods=['GET', 'POST'])
    def generate_report(filename):
        """Generate a report."""
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        
        if not os.path.exists(filepath):
            return render_template('error.html', message=f"File not found: {filename}")
        
        try:
            # Load the data
            if filename.endswith('.csv'):
                data = loader.load_csv(filepath)
            elif filename.endswith('.xlsx'):
                data = loader.load_excel(filepath)
            elif filename.endswith('.json'):
                data = loader.load_json(filepath)
            else:
                return render_template('error.html', message=f"Unsupported file type: {filename}")
            
            if request.method == 'POST':
                # Get form inputs
                title = request.form.get('title', f"Data Analysis Report: {filename}")
                format = request.form.get('format', 'html')
                
                # Set data for processor and visualizer
                processor.set_data(data)
                visualizer.set_data(data)
                
                # Create content blocks for report
                content_blocks = [
                    {'type': 'heading', 'content': title, 'level': 1},
                    {'type': 'heading', 'content': 'Data Summary', 'level': 2},
                    {'type': 'text', 'content': f"File: {filename}"},
                    {'type': 'text', 'content': f"Shape: {data.shape[0]} rows, {data.shape[1]} columns"},
                    {'type': 'text', 'content': f"Columns: {', '.join(data.columns)}"},
                    {'type': 'heading', 'content': 'Sample Data', 'level': 2},
                    {'type': 'dataframe', 'content': data.head(10), 'caption': 'First 10 rows of data'},
                    {'type': 'heading', 'content': 'Descriptive Statistics', 'level': 2},
                    {'type': 'dataframe', 'content': processor.describe(), 'caption': 'Descriptive statistics'}
                ]
                
                # Add visualizations
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    content_blocks.append({'type': 'heading', 'content': 'Visualizations', 'level': 2})
                    
                    # Correlation heatmap
                    if len(numeric_cols) > 1:
                        content_blocks.append({'type': 'heading', 'content': 'Correlation Matrix', 'level': 3})
                        fig = visualizer.heatmap(title='Correlation Matrix')
                        content_blocks.append({'type': 'figure', 'content': fig, 'caption': 'Correlation matrix of numeric columns'})
                    
                    # Histograms for numeric columns (limit to 5)
                    content_blocks.append({'type': 'heading', 'content': 'Distributions', 'level': 3})
                    for i, col in enumerate(numeric_cols[:5]):
                        fig = visualizer.histogram(column=col, title=f'Distribution of {col}')
                        content_blocks.append({'type': 'figure', 'content': fig, 'caption': f'Distribution of {col}'})
                
                # Generate report filename
                report_filename = f"report_{os.path.splitext(filename)[0]}.{format}"
                report_filepath = os.path.join(app.config['UPLOAD_FOLDER'], report_filename)
                
                # Export report
                if format == 'html':
                    exporter.export_html_report(title, content_blocks, report_filepath)
                else:  # pdf
                    exporter.export_pdf_report(title, content_blocks, report_filepath)
                
                # Provide download link
                download_url = url_for('download_file', filename=report_filename)
                
                return render_template('report_complete.html', report_filename=report_filename, 
                                      download_url=download_url)
            
            return render_template('report.html', filename=filename)
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return render_template('error.html', message=f"Error generating report: {str(e)}")
    
    @app.route('/download/<filename>')
    def download_file(filename):
        """Download a file."""
        return send_from_directory(app.config['UPLOAD_FOLDER'], secure_filename(filename), as_attachment=True)
    
    @app.route('/sample')
    def create_sample():
        """Create sample data for testing."""
        try:
            # Create sample data
            sample_data = create_sample_data(n_rows=1000, seed=42)
            
            # Save to CSV
            sample_filename = 'sample_data.csv'
            sample_filepath = os.path.join(app.config['UPLOAD_FOLDER'], sample_filename)
            sample_data.to_csv(sample_filepath, index=False)
            
            # Create sample stock data
            sample_stock = create_sample_stock_data(ticker='SAMPLE', n_days=252, seed=42)
            
            # Save to CSV
            stock_filename = 'sample_stock.csv'
            stock_filepath = os.path.join(app.config['UPLOAD_FOLDER'], stock_filename)
            sample_stock.to_csv(stock_filepath)
            
            return jsonify({
                'success': True,
                'message': 'Sample data created successfully',
                'files': [
                    {'name': sample_filename, 'url': url_for('data_view', filename=sample_filename)},
                    {'name': stock_filename, 'url': url_for('data_view', filename=stock_filename)}
                ]
            })
        
        except Exception as e:
            logger.error(f"Error creating sample data: {e}")
            return jsonify({'error': f'Error creating sample data: {str(e)}'}), 500
    
    @app.errorhandler(404)
    def page_not_found(e):
        """Handle 404 errors."""
        return render_template('error.html', message='Page not found'), 404
    
    @app.errorhandler(500)
    def server_error(e):
        """Handle 500 errors."""
        return render_template('error.html', message='Server error'), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)

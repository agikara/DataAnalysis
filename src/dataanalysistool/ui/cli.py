"""
Data Analysis Tool - UI Module - Command Line Interface

This module provides a command-line interface for the Data Analysis Tool.
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from .. import config
from ..data_loader import DataLoader
from ..data_processor import DataProcessor
from ..visualizer import Visualizer
from ..financial import FinancialAnalyzer
from ..exporter import Exporter
from ..utils import setup_logger

logger = logging.getLogger('dataanalysistool')

class DataAnalysisCLI:
    """
    Command-line interface for the Data Analysis Tool.
    """
    
    def __init__(self):
        """Initialize the CLI."""
        self.loader = DataLoader()
        self.processor = DataProcessor()
        self.visualizer = Visualizer()
        self.financial = FinancialAnalyzer()
        self.exporter = Exporter()
        
        # Set up parser
        self.parser = self._create_parser()
    
    def _create_parser(self):
        """
        Create the argument parser.
        
        Returns:
            argparse.ArgumentParser: The argument parser.
        """
        parser = argparse.ArgumentParser(
            description='Data Analysis Tool - A comprehensive Python-based data analysis tool'
        )
        
        # Add subparsers for commands
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # Load command
        load_parser = subparsers.add_parser('load', help='Load data from a file')
        load_parser.add_argument('--file', '-f', required=True, help='Path to the data file')
        load_parser.add_argument('--type', '-t', choices=['csv', 'excel', 'json'], default='csv',
                                help='File type (default: csv)')
        load_parser.add_argument('--sheet', '-s', help='Sheet name for Excel files')
        load_parser.add_argument('--output', '-o', help='Output file to save the loaded data')
        
        # Finance command
        finance_parser = subparsers.add_parser('finance', help='Perform financial analysis')
        finance_parser.add_argument('--ticker', '-t', required=True, help='Stock ticker symbol')
        finance_parser.add_argument('--period', '-p', default='1y',
                                  help='Period to download (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)')
        finance_parser.add_argument('--interval', '-i', default='1d',
                                  help='Data interval (1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo)')
        finance_parser.add_argument('--output', '-o', help='Output file to save the analysis results')
        finance_parser.add_argument('--format', '-f', choices=['html', 'pdf', 'excel', 'json'], default='html',
                                  help='Output format (default: html)')
        
        # Process command
        process_parser = subparsers.add_parser('process', help='Process data')
        process_parser.add_argument('--file', '-f', required=True, help='Path to the data file')
        process_parser.add_argument('--type', '-t', choices=['csv', 'excel', 'json'], default='csv',
                                  help='File type (default: csv)')
        process_parser.add_argument('--normalize', '-n', action='store_true', help='Normalize numeric columns')
        process_parser.add_argument('--missing', '-m', choices=['mean', 'median', 'mode', 'drop'],
                                  help='Strategy for handling missing values')
        process_parser.add_argument('--outliers', '-o', action='store_true', help='Remove outliers')
        process_parser.add_argument('--output', '-u', help='Output file to save the processed data')
        
        # Visualize command
        viz_parser = subparsers.add_parser('visualize', help='Create visualizations')
        viz_parser.add_argument('--file', '-f', required=True, help='Path to the data file')
        viz_parser.add_argument('--type', '-t', choices=['csv', 'excel', 'json'], default='csv',
                              help='File type (default: csv)')
        viz_parser.add_argument('--plot', '-p', choices=['line', 'bar', 'scatter', 'histogram', 'box', 'heatmap'],
                              required=True, help='Plot type')
        viz_parser.add_argument('--x', help='Column to use for x-axis')
        viz_parser.add_argument('--y', help='Column to use for y-axis')
        viz_parser.add_argument('--output', '-o', required=True, help='Output file to save the visualization')
        viz_parser.add_argument('--format', choices=['png', 'pdf', 'svg', 'jpg'], default='png',
                              help='Output format (default: png)')
        viz_parser.add_argument('--title', help='Plot title')
        
        # Report command
        report_parser = subparsers.add_parser('report', help='Generate a report')
        report_parser.add_argument('--file', '-f', required=True, help='Path to the data file')
        report_parser.add_argument('--type', '-t', choices=['csv', 'excel', 'json'], default='csv',
                                 help='File type (default: csv)')
        report_parser.add_argument('--output', '-o', required=True, help='Output file to save the report')
        report_parser.add_argument('--format', choices=['html', 'pdf'], default='html',
                                 help='Output format (default: html)')
        report_parser.add_argument('--title', help='Report title')
        
        # Dashboard command
        dashboard_parser = subparsers.add_parser('dashboard', help='Start the web dashboard')
        dashboard_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
        dashboard_parser.add_argument('--port', '-p', type=int, default=5000, help='Port to bind to (default: 5000)')
        dashboard_parser.add_argument('--debug', '-d', action='store_true', help='Enable debug mode')
        
        # Version command
        subparsers.add_parser('version', help='Show version information')
        
        return parser
    
    def run(self, args=None):
        """
        Run the CLI with the given arguments.
        
        Args:
            args (list, optional): Command-line arguments. If None, use sys.argv.
            
        Returns:
            int: Exit code.
        """
        # Parse arguments
        args = self.parser.parse_args(args)
        
        # Set up logging
        setup_logger()
        
        # Execute command
        if args.command == 'load':
            return self._handle_load(args)
        elif args.command == 'finance':
            return self._handle_finance(args)
        elif args.command == 'process':
            return self._handle_process(args)
        elif args.command == 'visualize':
            return self._handle_visualize(args)
        elif args.command == 'report':
            return self._handle_report(args)
        elif args.command == 'dashboard':
            return self._handle_dashboard(args)
        elif args.command == 'version':
            return self._handle_version(args)
        else:
            self.parser.print_help()
            return 0
    
    def _handle_load(self, args):
        """
        Handle the load command.
        
        Args:
            args: Command-line arguments.
            
        Returns:
            int: Exit code.
        """
        try:
            # Load data
            if args.type == 'csv':
                data = self.loader.load_csv(args.file)
            elif args.type == 'excel':
                data = self.loader.load_excel(args.file, sheet_name=args.sheet or 0)
            elif args.type == 'json':
                data = self.loader.load_json(args.file)
            else:
                logger.error(f"Unsupported file type: {args.type}")
                return 1
            
            # Print summary
            print(f"Loaded data from {args.file}")
            print(f"Shape: {data.shape}")
            print(f"Columns: {', '.join(data.columns)}")
            print("\nSample data:")
            print(data.head())
            
            # Save data if requested
            if args.output:
                if args.output.endswith('.csv'):
                    data.to_csv(args.output, index=False)
                elif args.output.endswith('.xlsx'):
                    data.to_excel(args.output, index=False)
                elif args.output.endswith('.json'):
                    data.to_json(args.output, orient='records')
                else:
                    data.to_csv(args.output, index=False)
                
                print(f"\nData saved to {args.output}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return 1
    
    def _handle_finance(self, args):
        """
        Handle the finance command.
        
        Args:
            args: Command-line arguments.
            
        Returns:
            int: Exit code.
        """
        try:
            # Load stock data
            print(f"Loading stock data for {args.ticker}...")
            data = self.loader.load_stock_data(args.ticker, period=args.period, interval=args.interval)
            
            # Set data for financial analyzer
            self.financial.set_data(data, ticker=args.ticker)
            
            # Calculate performance metrics
            print("Calculating performance metrics...")
            metrics = self.financial.calculate_performance_metrics()
            
            # Print summary
            print("\nPerformance Metrics:")
            print(f"Ticker: {metrics['ticker']}")
            print(f"Period: {metrics['start_date']} to {metrics['end_date']} ({metrics['total_days']} days)")
            print(f"Cumulative Return: {metrics['cumulative_return']:.2%}")
            print(f"Annualized Return: {metrics['annualized_return']:.2%}")
            print(f"Annualized Volatility: {metrics['annualized_volatility']:.2%}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Maximum Drawdown: {metrics['max_drawdown']:.2%}")
            print(f"Maximum Drawdown Duration: {metrics['max_drawdown_duration']} days")
            
            # Export results if requested
            if args.output:
                print(f"\nExporting results to {args.output}...")
                self.exporter.export_analysis_results(self.financial, args.ticker, format=args.format)
                print(f"Results exported to {args.output}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Error performing financial analysis: {e}")
            return 1
    
    def _handle_process(self, args):
        """
        Handle the process command.
        
        Args:
            args: Command-line arguments.
            
        Returns:
            int: Exit code.
        """
        try:
            # Load data
            if args.type == 'csv':
                data = self.loader.load_csv(args.file)
            elif args.type == 'excel':
                data = self.loader.load_excel(args.file)
            elif args.type == 'json':
                data = self.loader.load_json(args.file)
            else:
                logger.error(f"Unsupported file type: {args.type}")
                return 1
            
            # Set data for processor
            self.processor.set_data(data)
            
            # Process data
            if args.missing:
                if args.missing == 'drop':
                    data = data.dropna()
                    self.processor.set_data(data)
                    print("Dropped rows with missing values")
                else:
                    self.processor.handle_missing_values(strategy=args.missing)
                    print(f"Handled missing values using {args.missing} strategy")
            
            if args.normalize:
                self.processor.normalize()
                print("Normalized numeric columns")
            
            if args.outliers:
                self.processor.remove_outliers()
                print("Removed outliers")
            
            # Get processed data
            processed_data = self.processor.get_data()
            
            # Print summary
            print(f"\nProcessed data:")
            print(f"Shape: {processed_data.shape}")
            print(f"Columns: {', '.join(processed_data.columns)}")
            print("\nSample data:")
            print(processed_data.head())
            
            # Save data if requested
            if args.output:
                if args.output.endswith('.csv'):
                    processed_data.to_csv(args.output, index=False)
                elif args.output.endswith('.xlsx'):
                    processed_data.to_excel(args.output, index=False)
                elif args.output.endswith('.json'):
                    processed_data.to_json(args.output, orient='records')
                else:
                    processed_data.to_csv(args.output, index=False)
                
                print(f"\nProcessed data saved to {args.output}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return 1
    
    def _handle_visualize(self, args):
        """
        Handle the visualize command.
        
        Args:
            args: Command-line arguments.
            
        Returns:
            int: Exit code.
        """
        try:
            # Load data
            if args.type == 'csv':
                data = self.loader.load_csv(args.file)
            elif args.type == 'excel':
                data = self.loader.load_excel(args.file)
            elif args.type == 'json':
                data = self.loader.load_json(args.file)
            else:
                logger.error(f"Unsupported file type: {args.type}")
                return 1
            
            # Set data for visualizer
            self.visualizer.set_data(data)
            
            # Create visualization
            print(f"Creating {args.plot} plot...")
            
            if args.plot == 'line':
                fig = self.visualizer.line_plot(x=args.x, y=args.y, title=args.title)
            elif args.plot == 'bar':
                fig = self.visualizer.bar_plot(x=args.x, y=args.y, title=args.title)
            elif args.plot == 'scatter':
                if not args.x or not args.y:
                    logger.error("Scatter plot requires both x and y columns")
                    return 1
                fig = self.visualizer.scatter_plot(x=args.x, y=args.y, title=args.title)
            elif args.plot == 'histogram':
                if not args.x:
                    logger.error("Histogram requires a column")
                    return 1
                fig = self.visualizer.histogram(column=args.x, title=args.title)
            elif args.plot == 'box':
                fig = self.visualizer.box_plot(x=args.x, y=args.y, title=args.title)
            elif args.plot == 'heatmap':
                fig = self.visualizer.heatmap(title=args.title)
            else:
                logger.error(f"Unsupported plot type: {args.plot}")
                return 1
            
            # Save visualization
            self.visualizer.save_plot(fig, args.output, format=args.format)
            print(f"Visualization saved to {args.output}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return 1
    
    def _handle_report(self, args):
        """
        Handle the report command.
        
        Args:
            args: Command-line arguments.
            
        Returns:
            int: Exit code.
        """
        try:
            # Load data
            if args.type == 'csv':
                data = self.loader.load_csv(args.file)
            elif args.type == 'excel':
                data = self.loader.load_excel(args.file)
            elif args.type == 'json':
                data = self.loader.load_json(args.file)
            else:
                logger.error(f"Unsupported file type: {args.type}")
                return 1
            
            # Set data for processor and visualizer
            self.processor.set_data(data)
            self.visualizer.set_data(data)
            
            # Create content blocks for report
            title = args.title or f"Data Analysis Report: {os.path.basename(args.file)}"
            content_blocks = [
                {'type': 'heading', 'content': title, 'level': 1},
                {'type': 'heading', 'content': 'Data Summary', 'level': 2},
                {'type': 'text', 'content': f"File: {args.file}"},
                {'type': 'text', 'content': f"Shape: {data.shape[0]} rows, {data.shape[1]} columns"},
                {'type': 'text', 'content': f"Columns: {', '.join(data.columns)}"},
                {'type': 'heading', 'content': 'Sample Data', 'level': 2},
                {'type': 'dataframe', 'content': data.head(10), 'caption': 'First 10 rows of data'},
                {'type': 'heading', 'content': 'Descriptive Statistics', 'level': 2},
                {'type': 'dataframe', 'content': self.processor.describe(), 'caption': 'Descriptive statistics'}
            ]
            
            # Add visualizations
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                content_blocks.append({'type': 'heading', 'content': 'Visualizations', 'level': 2})
                
                # Correlation heatmap
                if len(numeric_cols) > 1:
                    content_blocks.append({'type': 'heading', 'content': 'Correlation Matrix', 'level': 3})
                    fig = self.visualizer.heatmap(title='Correlation Matrix')
                    content_blocks.append({'type': 'figure', 'content': fig, 'caption': 'Correlation matrix of numeric columns'})
                
                # Histograms for numeric columns (limit to 5)
                content_blocks.append({'type': 'heading', 'content': 'Distributions', 'level': 3})
                for i, col in enumerate(numeric_cols[:5]):
                    fig = self.visualizer.histogram(column=col, title=f'Distribution of {col}')
                    content_blocks.append({'type': 'figure', 'content': fig, 'caption': f'Distribution of {col}'})
            
            # Export report
            print(f"Generating {args.format} report...")
            if args.format == 'html':
                self.exporter.export_html_report(title, content_blocks, args.output)
            else:  # pdf
                self.exporter.export_pdf_report(title, content_blocks, args.output)
            
            print(f"Report saved to {args.output}")
            
            return 0
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return 1
    
    def _handle_dashboard(self, args):
        """
        Handle the dashboard command.
        
        Args:
            args: Command-line arguments.
            
        Returns:
            int: Exit code.
        """
        try:
            print(f"Starting web dashboard on http://{args.host}:{args.port}")
            print("Press Ctrl+C to stop")
            
            # Import web module here to avoid circular imports
            from .web import create_app
            
            app = create_app()
            app.run(host=args.host, port=args.port, debug=args.debug)
            
            return 0
        
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
            return 1
    
    def _handle_version(self, args):
        """
        Handle the version command.
        
        Args:
            args: Command-line arguments.
            
        Returns:
            int: Exit code.
        """
        try:
            # Get version from package metadata
            from importlib.metadata import version
            try:
                ver = version('dataanalysistool')
            except:
                ver = '0.1.0'  # Default version if not installed
            
            print(f"Data Analysis Tool v{ver}")
            print("A comprehensive Python-based data analysis tool")
            
            return 0
        
        except Exception as e:
            logger.error(f"Error getting version information: {e}")
            return 1

def main():
    """Main entry point for the CLI."""
    cli = DataAnalysisCLI()
    sys.exit(cli.run())

if __name__ == '__main__':
    main()

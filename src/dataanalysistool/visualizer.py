"""
Data Analysis Tool - Visualizer Module

This module provides functionality for data visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from .config import get_config

class Visualizer:
    """
    Class for creating data visualizations.
    """
    
    def __init__(self, data=None):
        """
        Initialize the Visualizer.
        
        Args:
            data (pandas.DataFrame, optional): Data to visualize.
        """
        self.data = data
        self.set_style()
    
    def set_data(self, data):
        """
        Set the data to visualize.
        
        Args:
            data (pandas.DataFrame): Data to visualize.
        """
        self.data = data
    
    def set_style(self, style=None):
        """
        Set the visualization style.
        
        Args:
            style (str, optional): Matplotlib/Seaborn style.
        """
        if style is None:
            style = get_config('default_theme')
        
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = get_config('default_figsize')
        plt.rcParams['figure.dpi'] = get_config('default_dpi')
    
    def line_plot(self, x=None, y=None, title=None, xlabel=None, ylabel=None, 
                 figsize=None, interactive=False, **kwargs):
        """
        Create a line plot.
        
        Args:
            x (str, optional): Column to use for x-axis.
            y (str or list, optional): Column(s) to use for y-axis.
            title (str, optional): Plot title.
            xlabel (str, optional): X-axis label.
            ylabel (str, optional): Y-axis label.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            **kwargs: Additional arguments to pass to the plotting function.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to visualize")
        
        if figsize is None:
            figsize = get_config('default_figsize')
        
        if interactive:
            # Create interactive plot with Plotly
            if x is None:
                # Use index as x if not specified
                plot_data = self.data.reset_index()
                x = 'index'
            else:
                plot_data = self.data
            
            if y is None:
                # Use all numeric columns if y not specified
                y = plot_data.select_dtypes(include=[np.number]).columns.tolist()
                if x in y:
                    y.remove(x)
            
            if isinstance(y, list):
                # Multiple y columns
                fig = px.line(plot_data, x=x, y=y, title=title, **kwargs)
            else:
                # Single y column
                fig = px.line(plot_data, x=x, y=y, title=title, **kwargs)
            
            if xlabel:
                fig.update_xaxes(title=xlabel)
            if ylabel:
                fig.update_yaxes(title=ylabel)
            
            return fig
        else:
            # Create static plot with Matplotlib
            fig, ax = plt.subplots(figsize=figsize)
            
            if x is None:
                # Use index as x if not specified
                if y is None:
                    # Use all numeric columns if y not specified
                    self.data.select_dtypes(include=[np.number]).plot(ax=ax, **kwargs)
                elif isinstance(y, list):
                    # Multiple y columns
                    self.data[y].plot(ax=ax, **kwargs)
                else:
                    # Single y column
                    self.data[y].plot(ax=ax, **kwargs)
            else:
                if y is None:
                    # Use all numeric columns except x if y not specified
                    y_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
                    if x in y_cols:
                        y_cols.remove(x)
                    self.data.plot(x=x, y=y_cols, ax=ax, **kwargs)
                elif isinstance(y, list):
                    # Multiple y columns
                    self.data.plot(x=x, y=y, ax=ax, **kwargs)
                else:
                    # Single y column
                    self.data.plot(x=x, y=y, ax=ax, **kwargs)
            
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            plt.tight_layout()
            return fig
    
    def bar_plot(self, x, y=None, title=None, xlabel=None, ylabel=None, 
                figsize=None, interactive=False, **kwargs):
        """
        Create a bar plot.
        
        Args:
            x (str): Column to use for x-axis.
            y (str, optional): Column to use for y-axis.
            title (str, optional): Plot title.
            xlabel (str, optional): X-axis label.
            ylabel (str, optional): Y-axis label.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            **kwargs: Additional arguments to pass to the plotting function.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to visualize")
        
        if figsize is None:
            figsize = get_config('default_figsize')
        
        if interactive:
            # Create interactive plot with Plotly
            if y is None:
                # Use count if y not specified
                fig = px.bar(self.data, x=x, title=title, **kwargs)
            else:
                fig = px.bar(self.data, x=x, y=y, title=title, **kwargs)
            
            if xlabel:
                fig.update_xaxes(title=xlabel)
            if ylabel:
                fig.update_yaxes(title=ylabel)
            
            return fig
        else:
            # Create static plot with Matplotlib
            fig, ax = plt.subplots(figsize=figsize)
            
            if y is None:
                # Use count if y not specified
                self.data[x].value_counts().sort_index().plot(kind='bar', ax=ax, **kwargs)
            else:
                self.data.plot(x=x, y=y, kind='bar', ax=ax, **kwargs)
            
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig
    
    def scatter_plot(self, x, y, title=None, xlabel=None, ylabel=None, 
                    figsize=None, interactive=False, **kwargs):
        """
        Create a scatter plot.
        
        Args:
            x (str): Column to use for x-axis.
            y (str): Column to use for y-axis.
            title (str, optional): Plot title.
            xlabel (str, optional): X-axis label.
            ylabel (str, optional): Y-axis label.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            **kwargs: Additional arguments to pass to the plotting function.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to visualize")
        
        if figsize is None:
            figsize = get_config('default_figsize')
        
        if interactive:
            # Create interactive plot with Plotly
            fig = px.scatter(self.data, x=x, y=y, title=title, **kwargs)
            
            if xlabel:
                fig.update_xaxes(title=xlabel)
            if ylabel:
                fig.update_yaxes(title=ylabel)
            
            return fig
        else:
            # Create static plot with Matplotlib
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.scatter(self.data[x], self.data[y], **kwargs)
            
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            plt.tight_layout()
            return fig
    
    def histogram(self, column, bins=10, title=None, xlabel=None, ylabel=None, 
                 figsize=None, interactive=False, **kwargs):
        """
        Create a histogram.
        
        Args:
            column (str): Column to plot.
            bins (int, optional): Number of bins.
            title (str, optional): Plot title.
            xlabel (str, optional): X-axis label.
            ylabel (str, optional): Y-axis label.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            **kwargs: Additional arguments to pass to the plotting function.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to visualize")
        
        if figsize is None:
            figsize = get_config('default_figsize')
        
        if interactive:
            # Create interactive plot with Plotly
            fig = px.histogram(self.data, x=column, nbins=bins, title=title, **kwargs)
            
            if xlabel:
                fig.update_xaxes(title=xlabel)
            if ylabel:
                fig.update_yaxes(title=ylabel)
            
            return fig
        else:
            # Create static plot with Matplotlib
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.hist(self.data[column], bins=bins, **kwargs)
            
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            plt.tight_layout()
            return fig
    
    def box_plot(self, x=None, y=None, title=None, xlabel=None, ylabel=None, 
                figsize=None, interactive=False, **kwargs):
        """
        Create a box plot.
        
        Args:
            x (str, optional): Column to use for x-axis.
            y (str, optional): Column to use for y-axis.
            title (str, optional): Plot title.
            xlabel (str, optional): X-axis label.
            ylabel (str, optional): Y-axis label.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            **kwargs: Additional arguments to pass to the plotting function.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to visualize")
        
        if figsize is None:
            figsize = get_config('default_figsize')
        
        if interactive:
            # Create interactive plot with Plotly
            if x is None and y is None:
                # Use all numeric columns
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                fig = px.box(self.data, y=numeric_cols, title=title, **kwargs)
            elif x is None:
                fig = px.box(self.data, y=y, title=title, **kwargs)
            elif y is None:
                fig = px.box(self.data, x=x, title=title, **kwargs)
            else:
                fig = px.box(self.data, x=x, y=y, title=title, **kwargs)
            
            if xlabel:
                fig.update_xaxes(title=xlabel)
            if ylabel:
                fig.update_yaxes(title=ylabel)
            
            return fig
        else:
            # Create static plot with Matplotlib
            fig, ax = plt.subplots(figsize=figsize)
            
            if x is None and y is None:
                # Use all numeric columns
                self.data.select_dtypes(include=[np.number]).boxplot(ax=ax, **kwargs)
            elif x is None:
                self.data[y].plot(kind='box', ax=ax, **kwargs)
            elif y is None:
                sns.boxplot(x=x, data=self.data, ax=ax, **kwargs)
            else:
                sns.boxplot(x=x, y=y, data=self.data, ax=ax, **kwargs)
            
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            plt.tight_layout()
            return fig
    
    def heatmap(self, data=None, title=None, figsize=None, interactive=False, **kwargs):
        """
        Create a heatmap.
        
        Args:
            data (pandas.DataFrame, optional): Data to plot. If None, use correlation matrix.
            title (str, optional): Plot title.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            **kwargs: Additional arguments to pass to the plotting function.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to visualize")
        
        if figsize is None:
            figsize = get_config('default_figsize')
        
        if data is None:
            # Use correlation matrix if data not specified
            data = self.data.select_dtypes(include=[np.number]).corr()
        
        if interactive:
            # Create interactive plot with Plotly
            fig = px.imshow(data, title=title, **kwargs)
            return fig
        else:
            # Create static plot with Matplotlib
            fig, ax = plt.subplots(figsize=figsize)
            
            sns.heatmap(data, ax=ax, annot=True, cmap='viridis', **kwargs)
            
            if title:
                ax.set_title(title)
            
            plt.tight_layout()
            return fig
    
    def pair_plot(self, columns=None, hue=None, title=None, figsize=None, **kwargs):
        """
        Create a pair plot.
        
        Args:
            columns (list, optional): Columns to include. If None, use all numeric columns.
            hue (str, optional): Column to use for color encoding.
            title (str, optional): Plot title.
            figsize (tuple, optional): Figure size.
            **kwargs: Additional arguments to pass to the plotting function.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to visualize")
        
        if columns is None:
            # Use all numeric columns if not specified
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        # Create pair plot with Seaborn
        g = sns.pairplot(self.data, vars=columns, hue=hue, **kwargs)
        
        if title:
            g.fig.suptitle(title, y=1.02)
        
        plt.tight_layout()
        return g
    
    def candlestick_chart(self, date_column=None, open_column='Open', high_column='High', 
                         low_column='Low', close_column='Close', volume_column='Volume',
                         title=None, figsize=None, **kwargs):
        """
        Create a candlestick chart for financial data.
        
        Args:
            date_column (str, optional): Column containing dates. If None, use index.
            open_column (str, optional): Column containing opening prices.
            high_column (str, optional): Column containing high prices.
            low_column (str, optional): Column containing low prices.
            close_column (str, optional): Column containing closing prices.
            volume_column (str, optional): Column containing volume data.
            title (str, optional): Plot title.
            figsize (tuple, optional): Figure size.
            **kwargs: Additional arguments to pass to the plotting function.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to visualize")
        
        # Prepare the data
        if date_column is None:
            # Use index as date if not specified
            df = self.data.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Index must be a DatetimeIndex when date_column is None")
        else:
            # Use specified date column
            df = self.data.copy()
            df['date'] = pd.to_datetime(df[date_column])
            df.set_index('date', inplace=True)
        
        # Create candlestick chart with Plotly
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df[open_column],
            high=df[high_column],
            low=df[low_column],
            close=df[close_column],
            name='Price'
        )])
        
        # Add volume as bar chart if available
        if volume_column in df.columns:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df[volume_column],
                name='Volume',
                yaxis='y2',
                marker_color='rgba(0,0,0,0.2)'
            ))
            
            # Set up secondary y-axis for volume
            fig.update_layout(
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right',
                    showgrid=False
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            **kwargs
        )
        
        return fig
    
    def save_plot(self, plot, filename, format='png', dpi=None):
        """
        Save a plot to a file.
        
        Args:
            plot: Plot object to save.
            filename (str): Filename to save to.
            format (str, optional): File format.
            dpi (int, optional): DPI for raster formats.
            
        Returns:
            str: Path to the saved file.
        """
        if dpi is None:
            dpi = get_config('default_dpi')
        
        if hasattr(plot, 'write_image'):
            # Plotly figure
            plot.write_image(filename, format=format)
        elif hasattr(plot, 'savefig'):
            # Matplotlib figure
            plot.savefig(filename, format=format, dpi=dpi)
        elif hasattr(plot, 'fig'):
            # Seaborn grid
            plot.fig.savefig(filename, format=format, dpi=dpi)
        else:
            raise ValueError("Unsupported plot type")
        
        return filename
    
    def plot_distribution(self, column, title=None, xlabel=None, ylabel=None, 
                         figsize=None, interactive=False, **kwargs):
        """
        Plot the distribution of a column.
        
        Args:
            column (str): Column to plot.
            title (str, optional): Plot title.
            xlabel (str, optional): X-axis label.
            ylabel (str, optional): Y-axis label.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            **kwargs: Additional arguments to pass to the plotting function.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to visualize")
        
        if figsize is None:
            figsize = get_config('default_figsize')
        
        if interactive:
            # Create interactive plot with Plotly
            fig = px.histogram(self.data, x=column, title=title, 
                              marginal='box', **kwargs)
            
            if xlabel:
                fig.update_xaxes(title=xlabel)
            if ylabel:
                fig.update_yaxes(title=ylabel)
            
            return fig
        else:
            # Create static plot with Matplotlib
            fig, ax = plt.subplots(figsize=figsize)
            
            sns.histplot(self.data[column], kde=True, ax=ax, **kwargs)
            
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            plt.tight_layout()
            return fig
    
    def plot_time_series(self, date_column=None, value_columns=None, title=None, 
                        xlabel=None, ylabel=None, figsize=None, interactive=False, **kwargs):
        """
        Plot time series data.
        
        Args:
            date_column (str, optional): Column containing dates. If None, use index.
            value_columns (str or list, optional): Column(s) to plot. If None, use all numeric columns.
            title (str, optional): Plot title.
            xlabel (str, optional): X-axis label.
            ylabel (str, optional): Y-axis label.
            figsize (tuple, optional): Figure size.
            interactive (bool, optional): Whether to create an interactive plot.
            **kwargs: Additional arguments to pass to the plotting function.
            
        Returns:
            The plot object.
        """
        if self.data is None:
            raise ValueError("No data to visualize")
        
        # Prepare the data
        if date_column is None:
            # Use index as date if not specified
            df = self.data.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Index must be a DatetimeIndex when date_column is None")
            x = df.index
        else:
            # Use specified date column
            df = self.data.copy()
            df[date_column] = pd.to_datetime(df[date_column])
            x = df[date_column]
        
        if value_columns is None:
            # Use all numeric columns if not specified
            value_columns = df.select_dtypes(include=[np.number]).columns
            if date_column in value_columns:
                value_columns = value_columns.drop(date_column)
        
        if figsize is None:
            figsize = get_config('default_figsize')
        
        if interactive:
            # Create interactive plot with Plotly
            if isinstance(value_columns, list):
                fig = px.line(df, x=x, y=value_columns, title=title, **kwargs)
            else:
                fig = px.line(df, x=x, y=value_columns, title=title, **kwargs)
            
            if xlabel:
                fig.update_xaxes(title=xlabel)
            if ylabel:
                fig.update_yaxes(title=ylabel)
            
            return fig
        else:
            # Create static plot with Matplotlib
            fig, ax = plt.subplots(figsize=figsize)
            
            if isinstance(value_columns, list):
                for col in value_columns:
                    ax.plot(x, df[col], label=col, **kwargs)
                ax.legend()
            else:
                ax.plot(x, df[value_columns], **kwargs)
            
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            return fig

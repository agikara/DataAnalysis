"""
Data Analysis Tool - Data Processor Module

This module provides functionality for processing and analyzing data.
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from .config import get_config

class DataProcessor:
    """
    Class for processing and analyzing data.
    """
    
    def __init__(self, data=None):
        """
        Initialize the DataProcessor.
        
        Args:
            data (pandas.DataFrame, optional): Data to process.
        """
        self.data = data
        self.original_data = data.copy() if data is not None else None
        self.transformations = []
    
    def set_data(self, data):
        """
        Set the data to process.
        
        Args:
            data (pandas.DataFrame): Data to process.
        """
        self.data = data
        self.original_data = data.copy()
        self.transformations = []
    
    def reset(self):
        """
        Reset the data to its original state.
        """
        if self.original_data is not None:
            self.data = self.original_data.copy()
            self.transformations = []
    
    def get_data(self):
        """
        Get the processed data.
        
        Returns:
            pandas.DataFrame: The processed data.
        """
        return self.data
    
    def get_transformations(self):
        """
        Get the list of applied transformations.
        
        Returns:
            list: The list of applied transformations.
        """
        return self.transformations
    
    def describe(self):
        """
        Generate descriptive statistics.
        
        Returns:
            pandas.DataFrame: Descriptive statistics.
        """
        if self.data is None:
            raise ValueError("No data to describe")
        
        return self.data.describe(include='all')
    
    def handle_missing_values(self, strategy='mean', columns=None):
        """
        Handle missing values in the data.
        
        Args:
            strategy (str, optional): Strategy for imputation ('mean', 'median', 'most_frequent', 'constant').
            columns (list, optional): Columns to process. If None, process all columns.
            
        Returns:
            pandas.DataFrame: The processed data.
        """
        if self.data is None:
            raise ValueError("No data to process")
        
        if columns is None:
            # Only process numeric columns by default
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        # Create a copy to avoid modifying the original
        result = self.data.copy()
        
        # Create imputer
        imputer = SimpleImputer(strategy=strategy)
        
        # Apply imputation
        result[columns] = imputer.fit_transform(result[columns])
        
        self.data = result
        self.transformations.append({
            'type': 'handle_missing_values',
            'strategy': strategy,
            'columns': list(columns)
        })
        
        return self.data
    
    def remove_duplicates(self, subset=None):
        """
        Remove duplicate rows from the data.
        
        Args:
            subset (list, optional): Columns to consider for identifying duplicates.
            
        Returns:
            pandas.DataFrame: The processed data.
        """
        if self.data is None:
            raise ValueError("No data to process")
        
        original_shape = self.data.shape
        self.data = self.data.drop_duplicates(subset=subset)
        
        self.transformations.append({
            'type': 'remove_duplicates',
            'subset': subset,
            'rows_removed': original_shape[0] - self.data.shape[0]
        })
        
        return self.data
    
    def filter_data(self, condition):
        """
        Filter data based on a condition.
        
        Args:
            condition: Boolean condition for filtering.
            
        Returns:
            pandas.DataFrame: The filtered data.
        """
        if self.data is None:
            raise ValueError("No data to filter")
        
        original_shape = self.data.shape
        self.data = self.data[condition]
        
        self.transformations.append({
            'type': 'filter_data',
            'rows_removed': original_shape[0] - self.data.shape[0]
        })
        
        return self.data
    
    def normalize(self, columns=None, method='minmax', range_min=0, range_max=1):
        """
        Normalize numeric columns in the data.
        
        Args:
            columns (list, optional): Columns to normalize. If None, normalize all numeric columns.
            method (str, optional): Normalization method ('minmax', 'zscore', 'robust').
            range_min (float, optional): Minimum value for minmax scaling.
            range_max (float, optional): Maximum value for minmax scaling.
            
        Returns:
            pandas.DataFrame: The processed data.
        """
        if self.data is None:
            raise ValueError("No data to normalize")
        
        if columns is None:
            # Only normalize numeric columns
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        # Create a copy to avoid modifying the original
        result = self.data.copy()
        
        if method == 'minmax':
            scaler = preprocessing.MinMaxScaler(feature_range=(range_min, range_max))
            result[columns] = scaler.fit_transform(result[columns])
        elif method == 'zscore':
            scaler = preprocessing.StandardScaler()
            result[columns] = scaler.fit_transform(result[columns])
        elif method == 'robust':
            scaler = preprocessing.RobustScaler()
            result[columns] = scaler.fit_transform(result[columns])
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.data = result
        self.transformations.append({
            'type': 'normalize',
            'method': method,
            'columns': list(columns)
        })
        
        return self.data
    
    def encode_categorical(self, columns=None, method='onehot'):
        """
        Encode categorical columns in the data.
        
        Args:
            columns (list, optional): Columns to encode. If None, encode all object columns.
            method (str, optional): Encoding method ('onehot', 'label', 'ordinal').
            
        Returns:
            pandas.DataFrame: The processed data.
        """
        if self.data is None:
            raise ValueError("No data to encode")
        
        if columns is None:
            # Only encode object columns
            columns = self.data.select_dtypes(include=['object', 'category']).columns
        
        # Create a copy to avoid modifying the original
        result = self.data.copy()
        
        if method == 'onehot':
            # One-hot encoding
            result = pd.get_dummies(result, columns=columns, drop_first=False)
        elif method == 'label':
            # Label encoding
            for col in columns:
                le = preprocessing.LabelEncoder()
                result[col] = le.fit_transform(result[col])
        elif method == 'ordinal':
            # Ordinal encoding (assumes categories are already in order)
            for col in columns:
                result[col] = result[col].astype('category').cat.codes
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        self.data = result
        self.transformations.append({
            'type': 'encode_categorical',
            'method': method,
            'columns': list(columns)
        })
        
        return self.data
    
    def detect_outliers(self, columns=None, method='zscore', threshold=3):
        """
        Detect outliers in the data.
        
        Args:
            columns (list, optional): Columns to check for outliers. If None, check all numeric columns.
            method (str, optional): Method for outlier detection ('zscore', 'iqr').
            threshold (float, optional): Threshold for outlier detection.
            
        Returns:
            pandas.DataFrame: DataFrame with outlier flags.
        """
        if self.data is None:
            raise ValueError("No data to process")
        
        if columns is None:
            # Only check numeric columns
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        outliers = pd.DataFrame(index=self.data.index)
        
        for col in columns:
            if method == 'zscore':
                # Z-score method
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outliers[f"{col}_outlier"] = z_scores > threshold
            elif method == 'iqr':
                # IQR method
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[f"{col}_outlier"] = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
        
        # Add a summary column
        outliers['is_outlier'] = outliers.any(axis=1)
        
        return outliers
    
    def remove_outliers(self, columns=None, method='zscore', threshold=3):
        """
        Remove outliers from the data.
        
        Args:
            columns (list, optional): Columns to check for outliers. If None, check all numeric columns.
            method (str, optional): Method for outlier detection ('zscore', 'iqr').
            threshold (float, optional): Threshold for outlier detection.
            
        Returns:
            pandas.DataFrame: The processed data.
        """
        if self.data is None:
            raise ValueError("No data to process")
        
        outliers = self.detect_outliers(columns, method, threshold)
        original_shape = self.data.shape
        
        # Remove rows with outliers
        self.data = self.data[~outliers['is_outlier']]
        
        self.transformations.append({
            'type': 'remove_outliers',
            'method': method,
            'threshold': threshold,
            'columns': list(columns) if columns is not None else 'all numeric',
            'rows_removed': original_shape[0] - self.data.shape[0]
        })
        
        return self.data
    
    def calculate_correlation(self, method='pearson'):
        """
        Calculate correlation matrix.
        
        Args:
            method (str, optional): Correlation method ('pearson', 'kendall', 'spearman').
            
        Returns:
            pandas.DataFrame: Correlation matrix.
        """
        if self.data is None:
            raise ValueError("No data to process")
        
        # Only include numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        return numeric_data.corr(method=method)
    
    def perform_pca(self, n_components=2, columns=None):
        """
        Perform Principal Component Analysis (PCA).
        
        Args:
            n_components (int, optional): Number of components to keep.
            columns (list, optional): Columns to include in PCA. If None, use all numeric columns.
            
        Returns:
            tuple: (transformed data, PCA object, explained variance ratio)
        """
        if self.data is None:
            raise ValueError("No data to process")
        
        if columns is None:
            # Only include numeric columns
            columns = self.data.select_dtypes(include=[np.number]).columns
        
        # Extract the data for PCA
        X = self.data[columns].values
        
        # Standardize the data
        X_std = preprocessing.StandardScaler().fit_transform(X)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_std)
        
        # Create a DataFrame with the principal components
        pca_df = pd.DataFrame(
            data=X_pca,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=self.data.index
        )
        
        return pca_df, pca, pca.explained_variance_ratio_
    
    def bin_data(self, column, bins=10, labels=None, strategy='uniform'):
        """
        Bin continuous data into discrete intervals.
        
        Args:
            column (str): Column to bin.
            bins (int or list, optional): Number of bins or bin edges.
            labels (list, optional): Labels for the bins.
            strategy (str, optional): Binning strategy ('uniform', 'quantile').
            
        Returns:
            pandas.Series: The binned data.
        """
        if self.data is None:
            raise ValueError("No data to process")
        
        if column not in self.data.columns:
            raise ValueError(f"Column not found: {column}")
        
        if strategy == 'uniform':
            # Uniform binning
            binned = pd.cut(self.data[column], bins=bins, labels=labels)
        elif strategy == 'quantile':
            # Quantile-based binning
            binned = pd.qcut(self.data[column], q=bins, labels=labels)
        else:
            raise ValueError(f"Unknown binning strategy: {strategy}")
        
        # Add the binned column to the data
        bin_column = f"{column}_binned"
        self.data[bin_column] = binned
        
        self.transformations.append({
            'type': 'bin_data',
            'column': column,
            'bins': bins if isinstance(bins, int) else len(bins) - 1,
            'strategy': strategy
        })
        
        return self.data[bin_column]
    
    def aggregate_data(self, group_by, agg_dict):
        """
        Aggregate data by group.
        
        Args:
            group_by (str or list): Column(s) to group by.
            agg_dict (dict): Dictionary mapping columns to aggregation functions.
            
        Returns:
            pandas.DataFrame: The aggregated data.
        """
        if self.data is None:
            raise ValueError("No data to aggregate")
        
        return self.data.groupby(group_by).agg(agg_dict)
    
    def pivot_data(self, index, columns, values, aggfunc='mean'):
        """
        Create a pivot table.
        
        Args:
            index (str or list): Column(s) to use as index.
            columns (str): Column to use as columns.
            values (str or list): Column(s) to aggregate.
            aggfunc (str or function, optional): Aggregation function.
            
        Returns:
            pandas.DataFrame: The pivot table.
        """
        if self.data is None:
            raise ValueError("No data to pivot")
        
        return pd.pivot_table(
            self.data,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc
        )
    
    def resample_time_series(self, date_column, rule, agg_dict=None):
        """
        Resample time series data.
        
        Args:
            date_column (str): Column containing dates.
            rule (str): Resampling rule (e.g., 'D', 'W', 'M').
            agg_dict (dict, optional): Dictionary mapping columns to aggregation functions.
            
        Returns:
            pandas.DataFrame: The resampled data.
        """
        if self.data is None:
            raise ValueError("No data to resample")
        
        # Set the date column as index if it's not already
        if self.data.index.name != date_column and date_column in self.data.columns:
            df = self.data.set_index(date_column)
        else:
            df = self.data.copy()
        
        # Ensure the index is datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
        
        # Resample the data
        if agg_dict is None:
            # Default to mean for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            agg_dict = {col: 'mean' for col in numeric_columns}
        
        resampled = df.resample(rule).agg(agg_dict)
        
        return resampled

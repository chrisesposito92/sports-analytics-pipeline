"""
Data processing module for sports analytics pipeline.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Base class for data processing."""
    
    def __init__(self, input_dir: str = "data/raw", output_dir: str = "data/processed"):
        """Initialize the data processor.
        
        Args:
            input_dir: Directory containing input data
            output_dir: Directory to save processed data
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from a CSV file.
        
        Args:
            filename: Name of the input file
            
        Returns:
            DataFrame containing the loaded data
        """
        filepath = os.path.join(self.input_dir, filename)
        return pd.read_csv(filepath)
    
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save processed data to a CSV file.
        
        Args:
            data: DataFrame to save
            filename: Name of the output file
        """
        filepath = os.path.join(self.output_dir, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the data.
        
        Args:
            data: DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        raise NotImplementedError("Subclasses must implement process()")


class SportsDataProcessor(DataProcessor):
    """Processor for sports data."""
    
    def clean_team_names(self, data: pd.DataFrame, team_col: str) -> pd.DataFrame:
        """Standardize team names.
        
        Args:
            data: DataFrame with team names
            team_col: Name of the column containing team names
            
        Returns:
            DataFrame with standardized team names
        """
        # Example team name standardization (would be expanded for real data)
        name_mapping = {
            "NY": "New York",
            "LA": "Los Angeles",
            "SF": "San Francisco",
            # Add more mappings as needed
        }
        
        # Create a copy to avoid SettingWithCopyWarning
        df = data.copy()
        
        # Apply standardization
        df[team_col] = df[team_col].replace(name_mapping)
        
        return df
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data.
        
        Args:
            data: DataFrame with missing values
            
        Returns:
            DataFrame with handled missing values
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = data.copy()
        
        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Fill missing numeric values with the mean of each column
        for col in numeric_cols:
            if df[col].isna().any():
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                logger.info(f"Filled {df[col].isna().sum()} missing values in {col} with mean {mean_val:.2f}")
        
        # Fill missing categorical values with the mode
        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            if df[col].isna().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Filled {df[col].isna().sum()} missing values in {col} with mode '{mode_val}'")
        
        return df
    
    def remove_duplicates(self, data: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Remove duplicate records from the data.
        
        Args:
            data: DataFrame that may contain duplicates
            subset: List of columns to consider when identifying duplicates
            
        Returns:
            DataFrame with duplicates removed
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = data.copy()
        
        # Count duplicates before removal
        n_before = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=subset, keep="first")
        
        # Log the number of duplicates removed
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(f"Removed {n_removed} duplicate records")
        
        return df
    
    def process(self, data: pd.DataFrame, home_team_col: str = "home_team", away_team_col: str = "away_team") -> pd.DataFrame:
        """Process sports data.
        
        Args:
            data: DataFrame to process
            home_team_col: Name of the column containing home team names
            away_team_col: Name of the column containing away team names
            
        Returns:
            Processed DataFrame
        """
        # Create a copy of the data
        df = data.copy()
        
        # Clean both home and away team names if these columns exist
        if home_team_col in df.columns:
            df = self.clean_team_names(df, home_team_col)
        
        if away_team_col in df.columns:
            df = self.clean_team_names(df, away_team_col)
            
        # Handle missing values and remove duplicates
        df = self.handle_missing_values(df)
        df = self.remove_duplicates(df)
        
        return df
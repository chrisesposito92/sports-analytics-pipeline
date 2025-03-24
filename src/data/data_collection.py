"""
Data collection module for sports analytics pipeline.
"""

import os
import requests
import pandas as pd
from typing import Dict, Any, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataCollector:
    """Base class for data collection."""
    
    def __init__(self, output_dir: str = "data/raw"):
        """Initialize the data collector.
        
        Args:
            output_dir: Directory to save collected data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def collect(self) -> pd.DataFrame:
        """Collect data and return as DataFrame."""
        raise NotImplementedError("Subclasses must implement collect()")
    
    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save collected data to file.
        
        Args:
            data: DataFrame to save
            filename: Name of the output file
        """
        filepath = os.path.join(self.output_dir, filename)
        data.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")


class APIDataCollector(DataCollector):
    """Collects data from a sports API."""
    
    def __init__(self, api_url: str, api_key: Optional[str] = None, output_dir: str = "data/raw"):
        """Initialize the API data collector.
        
        Args:
            api_url: URL of the API endpoint
            api_key: API key for authentication (if required)
            output_dir: Directory to save collected data
        """
        super().__init__(output_dir)
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a request to the API.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters for the request
            
        Returns:
            JSON response from the API
        """
        url = f"{self.api_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {url}: {e}")
            raise
    
    def collect(self, endpoint: str, params: Dict[str, Any] = None) -> pd.DataFrame:
        """Collect data from the API.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters for the request
            
        Returns:
            DataFrame containing the collected data
        """
        data = self._make_request(endpoint, params)
        return pd.json_normalize(data)


class CSVDataCollector(DataCollector):
    """Collects data from local CSV files."""
    
    def __init__(self, input_dir: str = "data/external", output_dir: str = "data/raw"):
        """Initialize the CSV data collector.
        
        Args:
            input_dir: Directory containing input CSV files
            output_dir: Directory to save processed data
        """
        super().__init__(output_dir)
        self.input_dir = input_dir
    
    def collect(self, filename: str) -> pd.DataFrame:
        """Read data from a CSV file.
        
        Args:
            filename: Name of the input CSV file
            
        Returns:
            DataFrame containing the data from the CSV file
        """
        filepath = os.path.join(self.input_dir, filename)
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            raise
"""
data_loading.py

This module provides a DataLoader class for loading CSV data
"""

import os
import pandas as pd
from utils.utils import csv_path


class DataLoader:
    """
    DataLoader class for loading and handling CSV data.
    """

    def __init__(self, filename):
        """
        Initializes the DataLoader with a specific CSV file.

        Parameters:
        - filename (str): Name of the CSV file to load.
        """
        self.file_path = os.path.join(csv_path, filename)
        self.data = None

    def load_data(self):
        """
        Loads the CSV file into a pandas DataFrame.

        Returns:
        - pd.DataFrame: The loaded dataset.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        self.data = pd.read_csv(self.file_path)
        #breakpoint()
        return self.data

    def get_data(self):
        """
        Returns the loaded data.

        Returns:
        - pd.DataFrame: The dataset if loaded, else raises an error.
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.data

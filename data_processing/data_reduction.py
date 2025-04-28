"""
data_reduction.py

This module contains a DataReducer class for balancing datasets by selecting 
a fixed number of samples from each arrhythmia class.
"""
import pandas as pd

class DataReducer:
    """
    A class to reduce dataset size by selecting a fixed number of samples per class.

    Attributes:
        df (DataFrame): The full dataset containing signals and labels.
        arrhythmia_classes (list): List of arrhythmia class labels.
        max_samples (int): Maximum number of samples to retain per class.
    """

    def __init__(self, df, arrhythmia_classes_cpsc, max_samples=400):
        """
        Initializes the DataReducer.

        Parameters:
            df (DataFrame): The full dataset with a 'classes' column.
            arrhythmia_classes (list): List of arrhythmia class labels to filter.
            max_samples (int): Maximum number of samples to retain per class (default=400).
        """
        self.df = df
        self.arrhythmia_classes_cpsc = arrhythmia_classes_cpsc
        self.max_samples = max_samples

    def reduce_data(self):
        """
        Reduces the dataset by selecting up to `max_samples` instances per class.

        Returns:
            DataFrame: The reduced dataset.
        """
        reduced_data_list = []
        
        for class_name in self.arrhythmia_classes_cpsc:
            class_subset = self.df[self.df['classes'] == class_name][:self.max_samples]
            reduced_data_list.append(class_subset)

        # Merge all subsets into a single DataFrame
        reduced_data = pd.concat(reduced_data_list, ignore_index=True)
        return reduced_data

"""
data_splitter.py

DataSplitter class 
for splitting data into training, validation, and test sets.

"""

from sklearn.model_selection import train_test_split

class DataSplitter:
    """
    DataSplitter class for splitting a dataset into training, validation, and test sets.
    """

    def __init__(self, data, train_size=0.75, val_size=0.15, test_size=0.10, random_state=42):
        """
        Initializes the DataSplitter with dataset and split parameters.

        Parameters:
        - data (pd.DataFrame): The dataset to split.
        - train_size (float): Proportion of the dataset to include in the train split (default: 75%).
        - val_size (float): Proportion of the dataset to include in the validation split (default: 15%).
        - test_size (float): Proportion of the dataset to include in the test split (default: 10%).
        - random_state (int): Random seed for reproducibility.
        """
        if data is None:
            raise ValueError("Data not loaded. Please provide a valid DataFrame.")

        assert train_size + val_size + test_size == 1, "Train, validation, and test sizes must sum to 1."

        self.data = data
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

    def split(self):
        """
        Splits the dataset into training, validation, and test sets.

        Returns:
        - tuple: (train_df, val_df, test_df) - Pandas DataFrames for train, validation, and test sets.
        """
        # First, split into train and temp (val + test)
        train_df, temp_df = train_test_split(
            self.data, train_size=self.train_size, random_state=self.random_state, stratify=self.data['classes']
        )

        # Split temp_df into validation and test sets
        val_df, test_df = train_test_split(
            temp_df, test_size=self.test_size / (self.val_size + self.test_size), random_state=self.random_state, stratify=temp_df['classes']
        )

        return train_df, val_df, test_df
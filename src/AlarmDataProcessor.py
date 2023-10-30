import numpy as np
import pandas as pd

class AlarmDataPreprocessor:
    def __init__(self, min_occurrence=3):
        """
        Initialize AlarmDataPreprocessor with the minimum occurrence threshold.
        
        Parameters:
            min_occurrence (int): Minimum occurrence for a row to be retained. Default is 3.
        """
        self.min_occurrence = min_occurrence
    
    def preprocess(self, alarm_data):
        """
        Preprocesses the alarm data as described in the original function.
        
        Parameters:
            alarm_data (pandas.DataFrame): The alarm data to be preprocessed. 
                                           Expected to have at least three columns.
        
        Returns:
            pandas.DataFrame: A new DataFrame containing the preprocessed alarm data, with the same column labels as the input.
        """
        # Convert DataFrame to NumPy array for efficient operations
        alarm_data_numpy = alarm_data.to_numpy()

        # Sort by first then second columns
        alarm_data_numpy = alarm_data_numpy[np.lexsort((alarm_data_numpy[:, 1], alarm_data_numpy[:, 0]))]

        num_rows = len(alarm_data_numpy)  # Number of rows in the data
        consecutive_count = 1  # Counter for identical consecutive rows
        reset_count = consecutive_count  # Variable to reset consecutive_count

        # Filter out rows that occur less than 'min_occurrence' times consecutively
        for i in range(num_rows - 1):
            if np.array_equal(alarm_data_numpy[i + 1, :2], alarm_data_numpy[i, :2]):
                consecutive_count += 1
            else:
                reset_count, consecutive_count = consecutive_count, 1

            # Edge case: last row comparison
            if i == num_rows - 2 and alarm_data_numpy[num_rows - 1, 1] != alarm_data_numpy[num_rows - 2, 1]:
                alarm_data_numpy[num_rows - 1, :] = 0

            # Remove rows that don't meet the threshold
            if reset_count < self.min_occurrence:
                for j in range(reset_count):
                    alarm_data_numpy[i - j, :] = 0
                reset_count = self.min_occurrence

        # Sort by columns
        alarm_data_numpy = alarm_data_numpy[np.lexsort((alarm_data_numpy[:, 2], alarm_data_numpy[:, 0]))]
        alarm_data_numpy = alarm_data_numpy[np.lexsort((alarm_data_numpy[:, 1], alarm_data_numpy[:, 0]))]

        # Remove zero rows
        alarm_data_numpy = alarm_data_numpy[~np.all(alarm_data_numpy == 0, axis=1)]

        # Convert back to DataFrame
        return pd.DataFrame(alarm_data_numpy, columns=alarm_data.columns)

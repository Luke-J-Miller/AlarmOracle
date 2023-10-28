
from numpy import numpy as np
import random
from pandas import pandas as pd

class AlarmDataGenerator:
    """Class for generating synthetic alarm data for training.
    Initialize the AlarmDataGenerator with specific attributes.

    Parameters:
    - num_alarm_ids: Total number of unique alarms.
    - num_device_ids: Total number of unique devices.
    - duration: Time duration for which to generate the synthetic alarm data.
    - random_alarm_prob: Probability of an alarm occurring randomly.
    - causal_alarm_prob: Probability of one alarm causing another alarm.
    - device_connectedness: Probability that one device is connected to another.
    - alarm_connectedness: Probability that one alarm is connected to another.
    - alarm_per_device_density: Density of alarms per device.
    - ground_truth_masking_level: Percentage of values to mask in the ground truth for training.
    """

    def __init__(self, num_alarm_ids=30, num_device_ids=50, duration=100_000, 
                 random_alarm_prob=1e-8, causal_alarm_prob=5e-4, 
                 device_connectedness=0.2, alarm_connectedness=0.2, 
                 alarm_per_device_density=0.5, ground_truth_masking_level=0.3):
        """Initialize class attributes."""
        self.num_alarm_ids = num_alarm_ids
        self.num_device_ids = num_device_ids
        self.duration = duration
        self.random_alarm_prob = random_alarm_prob
        self.causal_alarm_prob = causal_alarm_prob
        self.device_connectedness = device_connectedness
        self.alarm_connectedness = alarm_connectedness
        self.alarm_per_device_density = alarm_per_device_density
        self.ground_truth_masking_level = ground_truth_masking_level

    def generate_device_adjacency_matrix(self):
        """
        Generate an adjacency matrix for devices based on the device connectedness probability.

        Returns:
        - Numpy array representing the device adjacency matrix.
        """
        device_matrix = (np.random.rand(self.num_device_ids, self.num_device_ids) < self.device_connectedness).astype(int)
        np.fill_diagonal(device_matrix, 0)
        return device_matrix

    def generate_alarm_adjacency_matrix(self):
        """
        Generate an adjacency matrix for alarms based on the alarm connectedness probability.

        Returns:
        - Numpy array representing the alarm adjacency matrix.
        """
        adj_matrix = (np.random.rand(self.num_alarm_ids, self.num_alarm_ids) < self.alarm_connectedness).astype(int)
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix

    def map_alarms_to_devices(self):
        """
        Allocate specific alarms to specific devices based on the density of alarms per device.

        Returns:
        - Numpy array mapping alarms to devices.
        """
        alarm_to_device = (np.random.rand(self.num_device_ids, self.num_alarm_ids) < self.alarm_per_device_density).astype(int)
        for col in range(alarm_to_device.shape[1]):
            if np.sum(alarm_to_device[:, col]) == 0:
                random_device = np.random.randint(0, self.num_device_ids)
                alarm_to_device[random_device, col] = 1
        return alarm_to_device

    def mask_ground_truth(self, alarm_adj_matrix):
        """
        Hide some of the values in the ground truth alarm adjacency matrix.

        Parameters:
        - alarm_adj_matrix: The original alarm adjacency matrix.

        Returns:
        - Masked alarm adjacency matrix.
        """
        masked_matrix = np.copy(alarm_adj_matrix)
        for col in range(masked_matrix.shape[1]):
            for row in range(masked_matrix.shape[0]):
                if np.random.rand() < self.ground_truth_masking_level:
                    masked_matrix[row, col] = -1
        return masked_matrix

    def generate_alarm_data_and_matrix(self):
        """
        Generate the synthetic alarm data and adjacency matrices.

        Returns:
        - Dictionary containing the alarm DataFrame, alarm adjacency matrix,
          masked alarm adjacency matrix, and device adjacency matrix.
        """
        alarm_to_device = self.map_alarms_to_devices()
        device_adj_matrix = self.generate_device_adjacency_matrix()
        alarm_adj_matrix = self.generate_alarm_adjacency_matrix()
        alarm_data = []

        for t in range(self.duration):
            alarm_duration = int(3 * (1.1 ** random.randint(1, 50)))
            alarm_generated = False

            for alarm_id in range(self.num_alarm_ids):
                if alarm_generated:
                    break
                device_ids = np.where(alarm_to_device[:, alarm_id] == 1)[0]

                if np.random.rand() < self.random_alarm_prob:
                    alarm_data.append([alarm_id, random.choice(device_ids), t, t + alarm_duration])
                    alarm_generated = True
                    break

                for source_alarm_id in range(self.num_alarm_ids):
                    if alarm_generated:
                        break
                    if alarm_adj_matrix[source_alarm_id, alarm_id] and np.random.rand() < self.causal_alarm_prob:
                        alarm_data.append([alarm_id, random.choice(device_ids), t, t + alarm_duration])
                        alarm_generated = True
                        break

                for device_id in device_ids:
                    if alarm_generated:
                        break
                    adjacent_devices = np.where(device_adj_matrix[device_id] == 1)[0]
                    for adjacent_device in adjacent_devices:
                        if alarm_generated:
                            break
                        adjacent_alarms = np.where(alarm_to_device[adjacent_device, :] == 1)[0]
                        for adjacent_alarm in adjacent_alarms:
                            if np.random.rand() < self.causal_alarm_prob:
                                alarm_data.append([adjacent_alarm, adjacent_device, t, t + alarm_duration])
                                alarm_generated = True
                                break

        masked_alarm_adj_matrix = self.mask_ground_truth(alarm_adj_matrix)
        alarm_df = pd.DataFrame(alarm_data, columns=['alarm_id', 'device_id', 'start_timestamp', 'end_timestamp'])
        dataset_dict = {'alarm_df': alarm_df, 'alarm_adj_matrix': alarm_adj_matrix, 
                        'masked_alarm_adj_matrix': masked_alarm_adj_matrix, 
                        'device_adj_matrix': device_adj_matrix}

        return dataset_dict

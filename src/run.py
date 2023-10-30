import pandas as pd
import numpy as np
import os
from castle.common import GraphDAG  
from castle.metrics import MetricsDAG 
from castle.datasets import DAG, Topology  
import random
import logging

from NetworkAlarmGenerator.py import AlarmDataGenerator
from AlarmDataProcessor.py import AlarmDataPreprocessor
from TemporalProcessMiner.py import TemporalProcessMiner
from DataVisualization.py import Visualize

NUM_ALARM_DATASETS = 4  # Constant for the number of alarm datasets

def gen_dataset(randomly=True, params=None):
    """
    Generates alarm dataset using AlarmDataGenerator and preprocesses it.

    Parameters:
    ----------
    randomly: bool
        Whether to generate dataset with random parameters.
    params: list
        List of predefined parameters if not generating randomly.

    Returns:
    --------
    dataset_dict: dict
        Dictionary containing alarm data and relevant matrices.
    """
    # Initialize the AlarmDataGenerator
    generator = AlarmDataGenerator

    # Generate random parameters for AlarmDataGenerator if specified
    if randomly:
        num_alarm_ids = random.randint(20, 100)
        num_device_ids = random.randint(20, 100)
        duration = random.randint(20, 100)
        random_alarm_prob = random.uniform(1e-9, 1e-6)
        causal_alarm_prob = random.uniform(1e-8, 1e-4)
        device_connectedness = random.uniform(0.1, 0.5)
        alarm_connectedness = random.uniform(0.1, 0.5)
        alarm_per_device_density = random.random()
        ground_truth_masking_level = random.uniform(0.1, 0.5)
    else:
        num_alarm_ids, num_device_ids, duration, random_alarm_prob, \
        causal_alarm_prob, device_connectedness, alarm_connectedness, \
        alarm_per_device_density, ground_truth_masking_level = params
    # Instantiate AlarmDataGenerator with selected parameters
    generator = AlarmDataGenerator(num_alarm_ids, num_device_ids, duration, random_alarm_prob, 
                                   causal_alarm_prob, device_connectedness, alarm_connectedness, 
                                   alarm_per_device_density, ground_truth_masking_level)

    # Generate alarm data
    dataset_dict = generator.generate_alarm_data_and_matrix()
    
    # Preprocess alarm data
    preprocessor = AlarmDataPreprocessor()
    dataset_dict['alarm'] = preprocessor.preprocess(dataset_dict['alarm_df'])
    return dataset_dict  # Added this line to actually return the generated data


def model(alarm_data, topology_matrix):
    """
    Trains the model on the alarm data and returns the estimated causal matrix.

    Parameters:
    ----------
    alarm_data: pd.DataFrame
        Preprocessed alarm data.
    topology_matrix: np.ndarray
        Topology matrix of devices.

    Returns:
    --------
    est_causal_matrix: np.ndarray
        Estimated causal matrix.
    """
    # Prepare the input features
    X = alarm_data.iloc[:, 0:3]
    X.columns = ['event', 'node', 'timestamp']
    X = X.reindex(columns=['event', 'timestamp', 'node'])

    # Initialize directory for storing results
    base_dir = os.path.join('./3')
    
    # Initialize topology matrix if not provided
    if not topology_matrix.any():
        num_nodes = len(set(X['node']))
        topology_matrix = np.zeros((num_nodes, num_nodes))

    # Causal structure learning using TTPM
    ttpm = TTPM(topology_matrix, delta=0.01, max_hop=2, max_iter=50)
    ttpm.learn(X)
    
    # Obtain and return estimated causal structure
    est_causal_matrix = ttpm.causal_matrix.to_numpy()
    est_causal_matrix = (remove_diagnal_entries(ttpm.causal_matrix.values))
    return est_causal_matrix

def main():
    """
    Main function to generate datasets, train models, and visualize results.
    """
    datasets_dict = {}
    results_dict = {}
    
    # Generate datasets and train models
    for i in range(NUM_ALARM_DATASETS):
        dataset = gen_dataset()
        datasets_dict[f'dataset_{i}'] = dataset
        datasets_dict[f'dataset_{i}']['results'] = model(dataset['alarm_df'], dataset['device_adj_matrix'])
        
    # Visualize the results
    visualizer = Visualize()
    visualizer.visualize(datasets_dict)

if __name__ == '__main__':
    main()
    

import pandas as pd
import numpy as np
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, Topology
import random
import logging
from NetworkAlarmGenerator.py import AlarmDataGenerator
from AlarmDataProcessor.py import AlarmDataPreprocessor
from TemporalProcessMiner.py import TemporalProcessMiner
from DataVisualization.py import Visualize


NUM_ALARM_DATASETS = 4

def gen_dataset(randomly = True, params = None):
  generator = AlarmDataGenerator
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
  generator = AlarmDataGenerator(num_alarm_ids, num_device_ids, duration, random_alarm_prob, 
                                 causal_alarm_prob, device_connectedness, alarm_connectedness, 
                                 alarm_per_device_density, ground_truth_masking_level)
  dataset_dict = generator.generate_alarm_data_and_matrix()
  preprocessor = AlarmDataPreprocessor()
  dataset_dict['alarm'] = preprocessor.preprocess(dataset_dict['alarm_df'])

def model(alarm_data, topology_matrix):
    X = alarm_data.iloc[:,0:3]
    X.columns=['event','node','timestamp']
    X = X.reindex(columns=['event','timestamp','node'])
    base_dir = os.path.join('./3')
    if not topology_matrix.any():
        num_nodes = len(set(X['node']))
        topology_matrix = np.zeros((num_nodes, num_nodes))  

    # causal structure learning using TTPM
    ttpm = TTPM(topology_matrix,delta=0.01,max_hop=2,max_iter=50) 
    ttpm.learn(X)
    # Obtain estimated causal structure and save it
    est_causal_matrix = ttpm.causal_matrix.to_numpy()
    est_causal_matrix = (remove_diagnal_entries(ttpm.causal_matrix.values)) 
    return est_causal_matrix

def main:
  datasets_dict = {}
  results_dict = {}
  for i in range(NUM_ALARM_DATASETS):
    dataset = gen_dataset()
    datasets_dict[f'dataset_{i}'] = dataset
    datasets_dict[f'dataset_{i}']['results'] = model(dataset['alarm_df'], dataset['device_adj_matrix'])
  visualizer = Visualize()
  visualizer.visualize(datasets_dict)
    

import numpy as np  
import pandas as pd  
import networkx as nx
import Hawkes 

class Find_root_causes(object):
    """
    A class to identify root causes using the Hawkes model.

    Attributes:
    - num (int): The number of alarm data points to consider.
    - alarm_data (pd.DataFrame): The raw alarm data.
    - topology_matrix (numpy.ndarray): The matrix representing the topology.
    - est_causal_matrix (numpy.ndarray): The estimated causal relationship matrix.
    - causal_prior (numpy.ndarray): Prior information on causal relationships.
    - params (dict): Parameter settings for different types of data.

    Methods:
    - read: Reads and preprocesses alarm data.
    - run_model: Runs the Hawkes model and sets the estimated causal matrix.
    - main: The main function to run the overall procedure.
    """
    def __init__(self):
        self.num = 0  # Number of alarms to consider
        self.alarm_data = None  # Alarm data
        self.topology_matrix = None  # Topology matrix
        self.est_causal_matrix = None  # Estimated Causal matrix
        self.causal_prior = None  # Prior causal information
        # Parameter settings
        self.params = {
            1:{"time_decay": 0.015, "BIC_coef": 0.6, "sep": []},
            2:{"time_decay": 0.015, "BIC_coef": 0.6, "sep": []},
            6: {"time_decay": 0.015, "BIC_coef": 0.6, "sep": []},
            7: {"time_decay": 0.01, "BIC_coef": 0.5, "sep": [-11, -3]},
            8: {"time_decay": 0.015, "BIC_coef": 0.6, "sep": []},
            10: {"time_decay": 0.01, "BIC_coef": 1, "sep": [-18, -2]}
        }

    def read(self, k, count):
        """
        Reads and preprocesses alarm data based on the given number of alarms and a count limit.
        """
        self.num = k
        alarm_data = self.alarm_data
        # Filter and rename alarm data columns
        alarm_data = alarm_data.iloc[:, 0:3]
        alarm_data.columns = ['event', 'node', 'timestamp']
        self.alarm_data = alarm_data.reindex(columns=['event', 'timestamp', 'node'])
        if count == 0:
            self.alarm_data = self.alarm_data.iloc[:, :]
        else:
            self.alarm_data = self.alarm_data.iloc[:count, :]
        # Initialize topology matrix if it's not already done
        try:
            self.topology_matrix = self.topology_matrix
        except:
            self.topology_matrix = np.zeros(shape=(max(self.alarm_data["node"].values)+1, max(self.alarm_data["node"].values)+1))
        if self.num in [1, 2]:
            self.topology_matrix = np.zeros(
                shape=(max(self.alarm_data["node"].values) + 1, max(self.alarm_data["node"].values) + 1))

    def run_model(self, time_decay=0.1, max_links=1, model_selector='AIC', iters=100, BIC_coef=1.0, sep=[], prior_information=None):
        """
        Runs the Hawkes model to identify causal relationships.
        """
        if prior_information is None:
            model = TTPM(self.topology_matrix, max_links=max_links, iters=iters, time_decay=time_decay, model_selector=model_selector, BIC_coef=BIC_coef)
        # Initialize Hawkes model and learn
        else:
            model = Hawkes(self.topology_matrix, prior_information=prior_information, max_links=max_links, iters=iters, time_decay=time_decay, model_selector=model_selector, BIC_coef=BIC_coef)
        model.learn(self.alarm_data)
        # Get estimated causal matrix
        self.est_causal_matrix = model.causal_matrix.to_numpy()
        np.fill_diagonal(self.est_causal_matrix, 0)
        if sep:
            self.est_causal_matrix[sep[0]][sep[0]] = 1

    def main(self):
        """
        Main function to execute the analysis pipeline.
        """
        for i in range(1, 11):
            if i in [1, 2]:
                self.read(i, 5000)
                prior_information = self.causal_prior

                self.model(time_decay=self.params[i]["time_decay"], max_links=2, model_selector='BIC', iters=100,
                              BIC_coef=self.params[i]["BIC_coef"], sep=self.params[i]["sep"],prior_information=prior_information)

                self.est_causal_matrix[self.est_causal_matrix > 0] = 1
                return self.est_causal_matrix
        
            
            if i in [3, 4, 5, 9]:
                self.read(i, 0)
                self.model(time_decay=0.01, max_links=2, model_selector='BIC', iters=100, BIC_coef=1.0)
            
            if i in [6, 7, 8, 10]:
                self.read(i, 5000)
                self.model(time_decay=self.params[i]["time_decay"], max_links=2, model_selector='BIC', iters=100, BIC_coef=self.params[i]["BIC_coef"], sep=self.params[i]["sep"])

            

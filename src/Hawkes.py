import numpy as np
import pandas as pd
import networkx as nx
from itertools import product
import logging
#pip install gcastle
from castle.common import BaseLearner

class Hawkes(BaseLearner):
    """
    Hawkes class inherits from BaseLearner and implements the Hawkes process for learning causal structure.
    """
    def __init__(self, topology_matrix, prior_information, time_decay=0.1, 
                 BIC_coef=1, max_links=0, model_selector='BIC', iters=20):
        """
        Initialize the Hawkes object.
        
        Parameters:
        - topology_matrix: Numpy array representing the initial topology matrix.
        - prior_information: Array-like object containing prior knowledge of causal links.
        - time_decay: Float representing the time decay factor.
        - BIC_coef: Coefficient for BIC model selection.
        - max_links: Maximum number of links to consider for each event.
        - model_selector: String, either 'BIC' or 'AIC', for model selection criterion.
        - iters: Maximum number of iterations for hill-climbing.
        """
        BaseLearner.__init__(self)
        
        # Create a graph from topology_matrix using NetworkX
        self._topology = nx.from_numpy_array(topology_matrix, create_using=nx.Graph)
        
        # Initialize other instance variables
        self._model_selector = model_selector
        self._time_decay = time_decay
        self._max_links = max_links
        self._BIC_coef = BIC_coef
        self._iters = iters
        self._prior_information = prior_information

    def learn(self, alarm_data, *args, **kwargs):
        """
        Learn the causal structure from the given alarm data.
        
        Parameters:
        - alarm_data: DataFrame containing alarm event data.
        """
        self._gen_initial_values(alarm_data)
        _, raw_causal_matrix = self._climb_hill()
        self._causal_matrix = pd.DataFrame(raw_causal_matrix,
                                           index=self._matrix_names,
                                           columns=self._matrix_names)
        
    def _gen_initial_values(self, alarm_data):
        """
        Generates initial values required for subsequent methods in the class.
        
        Parameters:
        alarm_data (pd.DataFrame): A DataFrame containing alarm events, timestamps, and nodes.
        
        Attributes Set:
        self.alarm_data (pd.DataFrame): Filtered and modified alarm data.
        self._events (np.array): Unique events in alarm data.
        self._N (int): Number of unique events.
        self._matrix_names (list): List of unique event names, converted to strings.
        self._event_indexes (Series): Series mapping original event names to their index positions.
        self._g (NetworkX.Graph): Graph substructure based on nodes present in alarm_data.
        self._ne_grouped (DataFrameGroupBy): Grouped alarm data by node.
        self._decay_effects (np.array): Matrix to store decay effects.
        self._max_s_t (float): Maximum timestamp in alarm_data.
        self._min_s_t (float): Minimum timestamp in alarm_data.
        self._T (float): Product of the time range and the number of unique nodes.
        """
        
        # Remove any rows with missing values
        alarm_data.dropna(axis=0, how='any', inplace=True)
        
        # Convert timestamp to float type
        alarm_data['timestamp'] = alarm_data['timestamp'].astype(float)
        
        # Group data by 'event', 'timestamp', and 'node'; count the occurrences
        alarm_data = alarm_data.groupby(
            ['event', 'timestamp', 'node']).apply(len).reset_index()
        alarm_data.columns = ['event', 'timestamp', 'node', 'times']
        
        # Reorder columns for better readability
        alarm_data = alarm_data.reindex(columns=['node', 'timestamp', 'event', 'times'])
        
        # Sort data by 'node' and 'timestamp'
        alarm_data = alarm_data.sort_values(['node', 'timestamp'])
        
        # Filter data to only include nodes that are in the topology
        self.alarm_data = alarm_data[alarm_data['node'].isin(self._topology.nodes)]
        
        # Identify and sort unique events
        self._events = np.array(list(set(self.alarm_data['event'])))
        self._events.sort()
        
        # Count number of unique events
        self._N = len(self._events)
        
        # Generate list of event names
        self._matrix_names = list(self._events.astype(str))
        
        # Map event names to index values
        self._event_indexes = self._map_index(self.alarm_data['event'].values, self._events)
        self.alarm_data['event'] = self._event_indexes
        
        # Create subgraph based on nodes present in alarm_data
        self._g = self._topology.subgraph(self.alarm_data['node'].unique())
        
        # Group alarm data by node for future processing
        self._ne_grouped = self.alarm_data.groupby('node')
        
        # Initialize decay effects matrix
        self._decay_effects = np.zeros([len(self._events), self._max_links + 1])
        
        # Get maximum and minimum timestamps
        self._max_s_t = alarm_data['timestamp'].max()
        self._min_s_t = alarm_data['timestamp'].min()
        
        # Calculate decay effects
        for k in range(self._max_links + 1):
            self._decay_effects[:, k] = alarm_data.groupby('event').apply(
                lambda i: ((((1 - np.exp(
                    -self._time_decay * (self._max_s_t - i['timestamp']))) / self._time_decay)
                            * i['times']) * i['node'].apply(
                    lambda j: len(self._k_link_neibors(j, k)))).sum())
            
        # Calculate |V| x |T|, where |V| is the number of unique nodes and |T| is the time range
        self._T = (self._max_s_t - self._min_s_t) * len(alarm_data['node'].unique())

    def _k_link_neibors(self, node, k):
        """
        Returns the set of k-link neighbors for a given node in the graph.
        
        Parameters:
        node (str or int): The node for which k-link neighbors are to be found.
        k (int): The degree of separation between the given node and its k-link neighbors.
        
        Returns:
        set: A set containing the k-link neighbors of the given node.
        
        Notes:
        - When k is 0, the set will contain only the node itself.
        - The function uses Dijkstra's algorithm for finding the shortest paths.
        """
        
        if k == 0:
            # If k is 0, the only k-link neighbor is the node itself
            return {node}
        else:
            # Find all nodes reachable within a distance of 'k' using Dijkstra's algorithm
            nodes_within_k = set(nx.single_source_dijkstra_path_length(
                self._g, node, k).keys())
            
            # Find all nodes reachable within a distance of 'k-1' using Dijkstra's algorithm
            nodes_within_k_minus_1 = set(nx.single_source_dijkstra_path_length(
                self._g, node, k - 1).keys())
            
            # Return the set difference to get nodes that are exactly 'k' away from the given node
            return nodes_within_k - nodes_within_k_minus_1
        
    @staticmethod
    def _map_index(events, base_events):
        """
        Maps event names to their corresponding index values based on a list of base events.
        
        Parameters:
        events (numpy.ndarray): An array of event names that need to be mapped to index values.
        base_events (numpy.ndarray): An array of base event names against which the mapping will be performed.
        
        Returns:
        numpy.ndarray: An array containing the mapped index values.
        
        Notes:
        - This function assumes that all event names in 'events' exist in 'base_events'.
        - Uses numpy's 'where' function for efficient element-wise comparison.
        """
        
        # Map each event name in 'events' to its index in 'base_events'
        # 1. For each event in 'events', find where it matches in 'base_events'
        # 2. np.where() returns the indices where the condition is met.
        # 3. We assume the event exists and is unique, so take the first index ([0][0]).
        # 4. The resulting list is converted back to a numpy array.
        return np.array(list(map(lambda event: np.where(base_events == event)[0][0], events)))

    def _climb_hill(self):
        """
        Performs hill climbing to optimize the network model based on a likelihood function.
        
        Attributes Set in the Method:
        - self._prior_information: Prior information about the model, used to initialize the adjacency matrix.
        - self._N: Number of unique events.
        - self._iters: Number of iterations for hill climbing.
        
        Returns:
        tuple: Contains the optimal result and the corresponding adjacency matrix.
        
        Notes:
        - This method assumes that self._get_effect_decays() and self._param_search() are defined elsewhere.
        - Logging is used for tracking the progress.
        """
        
        # Calculate effect decays using a separate function.
        self._get_effect_decays()
        
        # Initialize the adjacency matrix 'edges' using prior information.
        # Add an identity matrix to keep the diagonal as ones.
        edges = ((self._prior_information.copy() == 1) + np.eye(self._N, self._N)).astype(int)
        
        # Perform an initial parameter search using the initialized adjacency matrix.
        # 'result' is assumed to be a tuple where the first element is a likelihood value.
        result = self._param_search(edges)
        l_ret = result[0]
        
        # Iterate for a pre-defined number of iterations
        for num_iter in range(self._iters):
            
            # Log the current likelihood score for tracking progress.
            logging.info('[iter {}]: likelihood_score = {}'.format(num_iter, l_ret))
            
            # 'stop_tag' indicates whether to stop the optimization or not.
            stop_tag = True
            
            # Iterate through one-step changes in the adjacency matrix.
            for new_edges in list(self._one_step_change_iterator(edges)):
                
                # Perform a parameter search with the new adjacency matrix.
                new_result = self._param_search(new_edges)
                new_l = new_result[0]
                
                # Check the termination condition: Stop if no improvement in likelihood.
                if new_l > l_ret:
                    result = new_result
                    l_ret = new_l
                    stop_tag = False
                    edges = new_edges

            # If the 'stop_tag' remains True, exit the loop.
            if stop_tag:
                return result, edges
        
        # Return the best result found.
        return result, edges

    def _get_effect_decays(self):
        """
        Initializes and populates the effect decays matrix, which stores the decaying effects of different links.
        
        Attributes Set in the Method:
        - self._max_links: Maximum number of links to be considered.
        - self.alarm_data: Data containing the alarms, assumed to be a pandas DataFrame.
        - self._events: List of unique events.
        
        Notes:
        - This method assumes that self._get_effect_decays_each_link() is defined elsewhere.
        - The effect decays matrix will be a 3D NumPy array.
        """
        
        # Initialize a 3D NumPy array to zero. Dimensions:
        #  - First: Maximum number of links + 1
        #  - Second: Number of rows in the alarm data
        #  - Third: Number of unique events
        self._effect_decays = np.zeros([self._max_links + 1,
                                        len(self.alarm_data),
                                        len(self._events)])

        # Loop through each link to populate the effect decays matrix
        for k in range(self._max_links + 1):
            self._get_effect_decays_each_link(k)

    def _get_effect_decays_each_link(self, k):
        """
        Populates the effect decays for a specific link (k) in self._effect_decays.

        Parameters:
        - k: The specific link index
        
        Attributes Set or Used:
        - self._N: Number of unique events
        - self.alarm_data: Data containing the alarms
        - self._time_decay: Decay rate for the temporal effect
        - self._effect_decays: 3D NumPy array storing the effect decays
        - self._ne_grouped: Alarm data grouped by nodes
        - self._k_link_neibors: Function to get k-link neighbors
        
        Notes:
        - This method updates self._effect_decays for the given link 'k'.
        """
        
        # Initialize variables
        j = 0  # index for looping through neighbor events
        pre_effect = np.zeros(self._N)  # previous effect decay
        alarm_data_array = self.alarm_data.values  # converting DataFrame to array for faster access

        # Loop through each row (alarm event) in the alarm data
        for item_ind in range(len(self.alarm_data)):
            sub_n, start_t, ala_i, times = alarm_data_array[item_ind, [0, 1, 2, 3]]  # unpacking current alarm event data
            last_sub_n, last_start_t, last_ala_i, last_times = alarm_data_array[item_ind - 1, [0, 1, 2, 3]]  # unpacking last alarm event data

            # Reset if we're looking at a new node or if timestamps are not ordered
            if (last_sub_n != sub_n) or (last_start_t > start_t):
                j = 0
                pre_effect = np.zeros(self._N)

                # Get the neighbors for the current node (sub_n) and link (k)
                try:
                    k_link_neighbors_ne = self._k_link_neibors(sub_n, k)
                    neighbors_table = pd.concat([self._ne_grouped.get_group(i) for i in k_link_neighbors_ne]).sort_values('timestamp')
                    neighbors_table_value = neighbors_table.values
                except ValueError as e:  # catch and ignore exceptions (e.g., no neighbors found)
                    k_link_neighbors_ne = []

                if len(k_link_neighbors_ne) == 0:
                    continue

            # Update the decay effect based on the last event's timing
            cur_effect = pre_effect * np.exp((np.min((last_start_t - start_t, 0))) * self._time_decay)

            # Loop through neighbors' events to update the current effect
            while True:
                try:
                    nei_sub_n, nei_start_t, nei_ala_i, nei_times = neighbors_table_value[j, :]
                except:  # break if out of bounds
                    break
                if nei_start_t < start_t:
                    cur_effect[int(nei_ala_i)] += nei_times * np.exp((nei_start_t - start_t) * self._time_decay)
                    j += 1
                else:
                    break

            # Update previous effect to current effect
            pre_effect = cur_effect

            # Update the effect decays for this specific link (k) and alarm event (item_ind)
            self._effect_decays[k, item_ind] = pre_effect

    def _param_search(self, edges):
        """
        Searches for the best parameters to maximize the likelihood function 
        of a given graphical model represented by the adjacency matrix.

        Parameters:
        - edges (numpy.ndarray): A square matrix representing the adjacency matrix 
        for the directed graph. A "1" in the (i, j)-th entry means there's a 
        directed edge from node i to node j.
        
        Returns:
        - tuple: A tuple containing the maximized likelihood score, optimized
        causal effects, and optimized causal rates.
        """
        
        # Convert numpy array to NetworkX DiGraph to check for cycles.
        causal_g = nx.from_numpy_array((edges - np.eye(self._N, self._N)), create_using=nx.DiGraph)

        # Check if the graph is a Directed Acyclic Graph (DAG).
        if not nx.is_directed_acyclic_graph(causal_g):
            return -1e14, np.zeros([len(self._events), len(self._events)]), np.zeros(len(self._events))

        # Initialize causal_effects and causal_rates (formerly mu) matrices.
        causal_effects = np.ones([self._max_links + 1, len(self._events), len(self._events)]) * edges
        causal_rates = np.ones(len(self._events))
        l_init = 0  # Initialize the total likelihood score.

        # Loop through each event to optimize its parameters.
        for i in range(len(self._events)):
            pa_i = set(np.where(edges[:, i] == 1)[0])  # Parents of the current event.
            li = -1e14  # Initialize the likelihood for the current event.
            
            # Get the times where the current event occurred.
            ind = np.where(self._event_indexes == i)
            x_i = self.alarm_data['times'].values[ind]
            x_i_all = np.zeros_like(self.alarm_data['times'].values)
            x_i_all[ind] = x_i
            
            while True:
                # Calculate the first term of the likelihood function.
                likelihood_i_sum = (self._decay_effects * causal_effects[:, :, i].T).sum() + causal_rates[i] * self._T

                # Initialize the likelihood for each alarm_data entry for the current event.
                likelihood_for_i = np.zeros(len(self.alarm_data)) + causal_rates[i]
                
                # Update the likelihoods based on the causal_effects.
                for k in range(self._max_links + 1):
                    likelihood_for_i += np.matmul(self._effect_decays[k, :], causal_effects[k, :, i].T)

                # Extract the likelihoods for the times the current event actually occurred.
                likelihood_for_i = likelihood_for_i[ind]
                
                # Calculate the second term of the likelihood function.
                x_log_likelihood = (x_i * np.log(likelihood_for_i)).sum()
                
                # Update the total likelihood for the current event.
                new_li = -likelihood_i_sum + x_log_likelihood

                # Convergence condition.
                if new_li - li < 0.1:
                    l_init += li  # Add the likelihood of the current event to the total likelihood.
                    break  # Break the while loop if the likelihood has converged.
                
                # Update the previous likelihood and causal_rates for the next iteration.
                li = new_li
                causal_rates[i] = ((causal_rates[i] / likelihood_for_i) * x_i).sum() / self._T

                # Update causal_effects based on the new likelihood and causal_rates.
                for j in pa_i:
                    for k in range(self._max_links + 1):
                        upper = ((causal_effects[k, j, i] * (self._effect_decays[k, :, j])[ind] / likelihood_for_i) * x_i).sum()
                        lower = self._decay_effects[j, k]
                        if lower == 0:
                            causal_effects[k, j, i] = 0
                            continue
                        causal_effects[k, j, i] = upper / lower

        # Apply either the AIC or BIC model selection criterion.
        if self._model_selector == 'AIC':
            return l_init - (len(self._events) + self._BIC_coef * edges.sum() * (self._max_links + 1)), causal_effects, causal_rates
        else:
            return l_init - (len(self._events) + self._BIC_coef * edges.sum() * (self._max_links + 1)) * np.log(
                self.alarm_data['times'].sum()) / 2, causal_effects, causal_rates
        
    @staticmethod
    def _one_step_change(edges, e):
        """
        Modify the adjacency matrix by flipping the state of a single directed edge.

        Given an edge (j, i), if there is a directed edge from node j to node i, remove it.
        If there is no directed edge from node j to node i, add one while making sure there
        isn't a reverse edge from node i to node j (to avoid cycles).

        Parameters:
        - edges (numpy.ndarray): A square matrix where edges[i, j] = 1 indicates a directed edge from node i to node j.
        - e (tuple): A tuple (j, i) indicating the edge to be flipped. j and i are the indices of the nodes connected by the edge.

        Returns:
        - numpy.ndarray: A new adjacency matrix with the specified edge flipped.
        """
        
        j, i = e  # Unpack the edge tuple
        
        # Don't change anything if it's a self-loop
        if j == i:
            return edges

        # Create a copy of the edges to modify
        new_edges = edges.copy()
        
        # If there is already an edge from node j to node i, remove it
        if new_edges[j, i] == 1:
            new_edges[j, i] = 0
            return new_edges
        else:
            # If there is no directed edge from node j to node i, add it.
            # Also, make sure to remove any reverse edge from i to j.
            new_edges[j, i] = 1
            new_edges[i, j] = 0
            return new_edges

import abc
import pandas as pd
import numpy as np
import networkx as nx
from itertools import product
from collections.abc import Iterable
from pandas import Index, RangeIndex

class BaseLearner(metaclass=abc.ABCMeta):
    """
    Abstract Base Class for implementing causal discovery learners.

    Attributes:
        _causal_matrix: numpy.ndarray or similar
            Stores the causal relationships discovered by the model.
    """

    def __init__(self):
        """
        Initialize a new instance of BaseLearner.
        """
        # Initialize _causal_matrix as None. Subclasses will populate this with real data.
        self._causal_matrix = None

    @abc.abstractmethod
    def learn(self, data, *args, **kwargs):
        """
        Abstract method to implement the causal discovery algorithm.

        Parameters:
            data: Data to learn from.
            *args, **kwargs: Additional positional and keyword arguments for flexibility.

        Raises:
            NotImplementedError: This is an abstract method and should be implemented in subclasses.
        """
        raise NotImplementedError

    @property
    def causal_matrix(self):
        """
        Property getter for the _causal_matrix attribute.

        Returns:
            The causal matrix stored in the object.
        """
        return self._causal_matrix

    @causal_matrix.setter
    def causal_matrix(self, value):
        """
        Property setter for the _causal_matrix attribute.

        Parameters:
            value: New value to set for the causal matrix.
        """
        self._causal_matrix = value

class AlarmOracle(BaseLearner):
    """

    A model that learns causal structures through temporal process mining with a Hawkes Process

    Parameters
    ----------
    topology_matrix: np.matrix
        Interpreted as an adjacency matrix to generate the graph.
        It should have two dimensions, and should be square.

    delta: float, default=0.1
            Time decaying coefficient for the exponential kernel.

    epsilon: int, default=1
        BIC penalty coefficient.

    max_hop: positive int, default=6
        The maximum considered hops in the topology,
        when ``max_hop=0``, it is divided by nodes, regardless of topology.

    penalty: str, default=BIC
        Two optional values: 'BIC' or 'AIC'.
        
    max_iter: int
        Maximum number of iterations.

    Examples
    --------
    >>> from castle.common import GraphDAG
    >>> from castle.metrics import MetricsDAG
    >>> X = alarm_data
    >>> model = AlarmOracle(topology_matrix, max_hop=2)
    >>> model.learn(X)
    >>> causal_matrix = ttpm.causal_matrix
    # plot est_dag and true_dag
    >>> GraphDAG(model.causal_matrix.values, true_causal_matrix)
    # calculate accuracy
    >>> ret_metrix = MetricsDAG(ttpm.causal_matrix.values, true_causal_matrix)
    >>> ret_metrix.metrics
    """

    def __init__(self, 
             topology_matrix, 
             delta=0.1, 
             epsilon=1,
             max_hop=0, 
             penalty='BIC', 
             max_iter=20):
        """
        Initialize the class with specified parameters.

        Parameters:
            topology_matrix (numpy.ndarray): A square adjacency matrix representing the topology of the network.
            delta (float, optional): A hyperparameter, default is 0.1.
            epsilon (float, optional): Another hyperparameter, default is 1.
            max_hop (int, optional): The maximum number of hops between nodes in the network, default is 0.
            penalty (str, optional): The penalty term to be used, default is 'BIC'.
            max_iter (int, optional): The maximum number of iterations for the learning process, default is 20.

        Raises:
            AssertionError: If the topology_matrix is not a square 2D numpy array.
        """
        # Call the constructor of the parent class
        BaseLearner.__init__(self)

        # Validate the topology matrix
        assert isinstance(topology_matrix, np.ndarray),\
            'topology_matrix should be np.ndarray object'
        assert topology_matrix.ndim == 2,\
            'topology_matrix should be two-dimensional'
        assert topology_matrix.shape[0] == topology_matrix.shape[1],\
            'The topology_matrix should be square.'

        # Initialize instance variables
        # Convert numpy matrix to a networkx graph object for easier manipulation
        self._topo = nx.from_numpy_array(topology_matrix,
                                         create_using=nx.Graph)

        # Initialize other instance variables
        self._penalty = penalty
        self._delta = delta
        self._max_hop = max_hop
        self._epsilon = epsilon
        self._max_iter = max_iter

    def learn(self, tensor, *args, **kwargs):
        """
        Implements the Temporal Tensor-based Process Mining (TTPM) algorithm to learn causal relationships.

        The function accepts a tensor in the form of a pandas DataFrame and applies the TTPM algorithm
        to generate a causal matrix (Directed Acyclic Graph, or DAG) that models the relationships 
        between different events happening at topological nodes over time.

        Parameters
        ----------
        tensor : pd.DataFrame
            A DataFrame containing event logs, expected to have the following three columns:

            - 'event': Specifies the type or name of the event.
            - 'timestamp': The timestamp at which the event occurred. 
                           Format should be Unix timestamp, e.g., '1615962101.0'.
            - 'node': Identifies the topological node where the event occurred.

        Raises
        ------
        TypeError
            Raised if the input tensor is not of type pd.DataFrame.
        ValueError
            Raised if the input tensor DataFrame does not contain expected columns.

        Notes
        -----
        The function initializes necessary variables using self._start_init() and then performs
        causal discovery through self._hill_climb(). The learned causal matrix is stored 
        in self._causal_matrix.
        """

        # Validate the data type of the tensor
        if not isinstance(tensor, pd.DataFrame):
            raise TypeError('Invalid tensor type: Only pd.DataFrame is supported at the moment.')

        # Validate the presence of required columns in the tensor DataFrame
        required_columns = ['event', 'timestamp', 'node']
        for col in required_columns:
            if col not in tensor.columns:
                raise ValueError(f"Missing required column: {col} in the tensor DataFrame.")

        # Initialize algorithm-specific variables
        self._start_init(tensor)

        # Apply the hill climbing algorithm to generate the raw causal matrix
        _, raw_causal_matrix = self._hill_climb()

        # Convert the raw causal matrix to a pandas DataFrame for better readability and manipulation
        self._causal_matrix = pd.DataFrame(raw_causal_matrix,
                                           index=self._matrix_names,
                                           columns=self._matrix_names)

    def _start_init(self, tensor):
        """
        Initializes necessary variables and pre-processes the input tensor for the TTPM algorithm.

        This function performs several key tasks:
        1. Drops NA values and converts timestamp to float.
        2. Aggregates data based on event, timestamp, and node.
        3. Sorts the tensor based on node and timestamp.
        4. Filters the tensor to only contain nodes that are in the topology.
        5. Computes the decay effects based on the events.

        Parameters
        ----------
        tensor : pd.DataFrame
            The input tensor DataFrame with columns: ['event', 'timestamp', 'node'].

        """

        # Drop NA values and convert the timestamp column to float
        tensor.dropna(axis=0, how='any', inplace=True)
        tensor['timestamp'] = tensor['timestamp'].astype(float)

        # Aggregate rows with the same ['event', 'timestamp', 'node']
        # This will count how many times the same event happens at the same node and timestamp
        tensor = tensor.groupby(['event', 'timestamp', 'node']).apply(len).reset_index()
        tensor.columns = ['event', 'timestamp', 'node', 'times']
        tensor = tensor.reindex(columns=['node', 'timestamp', 'event', 'times'])

        # Sort by node and timestamp
        tensor = tensor.sort_values(['node', 'timestamp'])

        # Filter out nodes that are not in the topology graph
        self.tensor = tensor[tensor['node'].isin(self._topo.nodes)]

        # Generate a list of unique event names, and sort it
        self._event_names = np.array(list(set(self.tensor['event'])))
        self._event_names.sort()

        # Initialize matrix dimension and names
        self._N = len(self._event_names)
        self._matrix_names = list(self._event_names.astype(str))

        # Map event names to index values for easier manipulation
        self._event_indexes = self._map_event_to_index(self.tensor['event'].values, self._event_names)
        self.tensor['event'] = self._event_indexes

        # Create a subgraph containing only nodes that exist in the tensor
        self._g = self._topo.subgraph(self.tensor['node'].unique())

        # Group the tensor by node for later use
        self._ne_grouped = self.tensor.groupby('node')

        # Initialize decay effects matrix
        self._decay_effects = np.zeros([len(self._event_names), self._max_hop + 1])

        # Calculate minimum and maximum timestamps
        self._max_s_t = tensor['timestamp'].max()
        self._min_s_t = tensor['timestamp'].min()

        # Compute decay effects based on timestamps and events
        for k in range(self._max_hop + 1):
            self._decay_effects[:, k] = tensor.groupby('event').apply(
                lambda i: ((((1 - np.exp(-self._delta * (self._max_s_t - i['timestamp']))) / self._delta) 
                            * i['times']) * i['node'].apply(lambda j: len(self._k_hop_neibors(j, k)))).sum())

        # Calculate total time slots multiplied by the number of unique nodes (|V| x |T|)
        self._T = (self._max_s_t - self._min_s_t) * len(tensor['node'].unique())


    def _k_hop_neighbors(self, node, k):
        """
        Returns the k-hop neighbors of a given node in the graph.

        This method identifies the set of nodes that are exactly k hops away from 
        the input node, using Dijkstra's algorithm for path length computation.

        Parameters
        ----------
        node : Any hashable type
            The node for which k-hop neighbors are to be found.

        k : int
            The number of hops away from the node to define the neighborhood.

        Returns
        -------
        set
            A set of nodes that are exactly k hops away from the input node.

        Notes
        -----
        - If k is 0, it returns a set containing only the input node itself.
        - The function uses NetworkX's single_source_dijkstra_path_length to find paths.

        """

        # If k is 0, the only node at 0 hops away is the node itself
        if k == 0:
            return {node}
        else:
            # Use Dijkstra's algorithm to find nodes at distance k
            nodes_at_distance_k = set(nx.single_source_dijkstra_path_length(self._g, node, k).keys())

            # Use Dijkstra's algorithm to find nodes at distance k - 1
            nodes_at_distance_k_minus_1 = set(nx.single_source_dijkstra_path_length(self._g, node, k - 1).keys())

            # Subtract to get nodes that are exactly k hops away
            return nodes_at_distance_k - nodes_at_distance_k_minus_1

    @staticmethod
    def _map_event_to_index(event_names, base_event_names):
        """
        Maps each event name in the given array to its corresponding index in a base array of event names.

        This method takes an array of event names (`event_names`) and maps each event to its corresponding
        index in the base array (`base_event_names`) using NumPy's where function. 

        Parameters
        ----------
        event_names : np.ndarray
            An array of event names, sorted by node and timestamp. Shape is generally (n,), where n is the number of events.

        base_event_names : np.ndarray
            An array containing unique, sorted event names. Shape is generally (m,), where m is the number of unique events.

        Returns
        -------
        np.ndarray
            An array where each event in `event_names` is replaced by its index in `base_event_names`. The shape is the same as `event_names`.

        Example
        -------
        >>> _map_event_to_index(np.array(['A', 'B', 'A']), np.array(['A', 'B', 'C']))
        np.array([0, 1, 0])

        """

        # Use lambda function to map each event name to its index in base_event_names.
        # np.where finds the index where the event_name matches in base_event_names and we take the first match ([0][0]).
        # This is then wrapped in a list comprehension to apply the lambda function to each element in event_names.
        return np.array([np.where(base_event_names == event_name)[0][0] for event_name in event_names])


    def _hill_climb(self):
        """
        Executes a hill climbing algorithm to find the best causal graph and subsequently generates the causal matrix (DAG).

        This method initializes an adjacency matrix and iteratively refines it by comparing the likelihood scores of potential new matrices.
        The iteration stops when no better (higher likelihood) adjacency matrix is found or when the maximum number of iterations is reached.

        Returns
        -------
        result : tuple
            Contains the following elements:
                - likelihood: float, serves as the score criteria for evaluating the causal structure.
                - alpha matrix: np.ndarray, represents the intensity of the causal effect from event v' to v.
                - events vector: np.ndarray, represents the exogenous base intensity of each event.

        edge_mat : np.ndarray
            The causal matrix (DAG) inferred from the data.

        Notes
        -----
        The internal methods `_get_effect_tensor_decays` and `_em` are assumed to be implemented and affect the behavior of this method.
        """

        # Initialize decay effects for the tensor
        self._get_effect_tensor_decays()

        # Initialize the adjacency matrix with identity matrix.
        # Here, `_N` is assumed to be the number of events.
        edge_mat = np.eye(self._N, self._N)

        # Calculate initial likelihood and parameters
        result = self._em(edge_mat)
        l_ret = result[0]

        # Start the hill climbing algorithm
        for num_iter in range(self._max_iter):

            # Log the current iteration and likelihood
            logging.info(f'[iter {num_iter}]: likelihood_score = {l_ret}')

            # Flag to determine if any improvement happened in this iteration
            stop_tag = True

            # Iterate over all possible single-step changes to the adjacency matrix
            for new_edge_mat in list(self._one_step_change_iterator(edge_mat)):

                # Evaluate the new adjacency matrix
                new_result = self._em(new_edge_mat)
                new_l = new_result[0]

                # If the new likelihood is greater, update the result and likelihood
                if new_l > l_ret:
                    result = new_result
                    l_ret = new_l
                    stop_tag = False  # An improvement was made
                    edge_mat = new_edge_mat

            # Termination condition: no improvement
            if stop_tag:
                return result, edge_mat

        # If max iterations reached, return the current best result
        return result, edge_mat

    def _get_effect_tensor_decays(self):
        """
        Initializes and populates the `_effect_tensor_decays` attribute.

        This method calculates the effect tensor decays for each hop up to `_max_hop` and stores them in the `_effect_tensor_decays` attribute.

        Notes
        -----
        The internal method `_get_effect_tensor_decays_each_hop` is assumed to be implemented and directly affects the behavior of this method.
        `_max_hop` is the maximum number of hops to be considered.
        `_effect_tensor_decays` is a 3D NumPy array where each slice along the first dimension represents a different hop.

        Attributes Modified
        -------------------
        _effect_tensor_decays : np.ndarray
            3D tensor holding the decay effects for each hop, each tensor data, and each event name.
        """

        # Initialize the 3D tensor to store the effect decays.
        # Dimension 1: Number of hops + 1 (from 0 to _max_hop)
        # Dimension 2: Number of tensor data points
        # Dimension 3: Number of unique events (_event_names)
        self._effect_tensor_decays = np.zeros([self._max_hop + 1,
                                               len(self.tensor),
                                               len(self._event_names)])

        # Loop through each hop to populate the effect tensor decays
        for k in range(self._max_hop + 1):
            # Calculate the effect tensor decays for each hop and store it in _effect_tensor_decays.
            # The actual computation is handled by _get_effect_tensor_decays_each_hop.
            self._get_effect_tensor_decays_each_hop(k)


    def _get_effect_tensor_decays_each_hop(self, k):
        """
        Computes the decay effects for events at a particular 'k' hop distance.

        For a given hop 'k', this method populates the `_effect_tensor_decays` attribute
        at the corresponding slice along the first dimension.

        Parameters
        ----------
        k : int
            The hop number to be considered for calculating the decay effects.

        Attributes Modified
        -------------------
        _effect_tensor_decays : np.ndarray
            Slice along the first dimension updated to include decay effects for the given hop 'k'.

        Notes
        -----
        Assumes that `_k_hop_neibors`, `_ne_grouped`, `_delta`, and other instance attributes are already initialized.
        """

        # Initialize the index for the neighbors_table
        j = 0

        # Initialize a vector to store the effect for each event type
        pre_effect = np.zeros(self._N)

        # Convert the tensor DataFrame to a NumPy array for faster access
        tensor_array = self.tensor.values

        # Loop through each item (event) in the tensor to compute its effect
        for item_ind in range(len(self.tensor)):

            # Extract relevant fields from the current tensor item
            sub_n, start_t, ala_i, times = tensor_array[item_ind, [0, 1, 2, 3]]

            # Extract relevant fields from the last tensor item for comparison
            last_sub_n, last_start_t, last_ala_i, last_times = tensor_array[item_ind - 1, [0, 1, 2, 3]]

            # Check conditions to reset 'j' and 'pre_effect'
            if (last_sub_n != sub_n) or (last_start_t > start_t):
                j = 0
                pre_effect = np.zeros(self._N)

                # Try to retrieve the k-hop neighbors of the current node
                try:
                    k_hop_neighbors_ne = self._k_hop_neibors(sub_n, k)
                    neighbors_table = pd.concat([self._ne_grouped.get_group(i) for i in k_hop_neighbors_ne])
                    neighbors_table = neighbors_table.sort_values('timestamp')
                    neighbors_table_value = neighbors_table.values
                except ValueError as e:
                    k_hop_neighbors_ne = []

                # If no k-hop neighbors, skip to the next iteration
                if len(k_hop_neighbors_ne) == 0:
                    continue

            # Compute the current effect
            cur_effect = pre_effect * np.exp((np.min((last_start_t - start_t, 0))) * self._delta)

            # Update 'cur_effect' based on the contribution of each k-hop neighbor
            while 1:
                try:
                    nei_sub_n, nei_start_t, nei_ala_i, nei_times = neighbors_table_value[j, :]
                except:
                    break

                if nei_start_t < start_t:
                    cur_effect[int(nei_ala_i)] += nei_times * np.exp((nei_start_t - start_t) * self._delta)
                    j += 1
                else:
                    break

            # Update 'pre_effect' for the next iteration
            pre_effect = cur_effect

            # Store the computed effect in '_effect_tensor_decays'
            self._effect_tensor_decays[k, item_ind] = pre_effect


    def _em(self, edge_mat):
        """
        The Expectation-Maximization (E-M) module for parameter optimization.

        This function uses the E-M algorithm to find the optimal parameters of a point process model.
        It iteratively refines the parameters to maximize the likelihood of observing the given events.

        Parameters
        ----------
        edge_mat : np.ndarray
            The adjacency matrix representing the causal relationships among events.

        Returns
        -------
        likelihood : float
            The likelihood of the data given the model. Used as the score criterion for selecting the causal structure.
        alpha : np.ndarray
            A tensor representing the intensity of causal effects from one event to another.
        mu : np.ndarray
            A vector representing the exogenous base intensity of each event.

        Attributes Modified
        -------------------
        None

        Notes
        -----
        Assumes the `_max_hop`, `_event_names`, `_event_indexes`, `_T`, `_decay_effects`, `_effect_tensor_decays`, `_penalty`, and `_epsilon` attributes are already initialized.
        """

        # Create a directed graph from the adjacency matrix, excluding self-loops
        causal_g = nx.from_numpy_array((edge_mat - np.eye(self._N, self._N)), create_using=nx.DiGraph)

        # Return extreme negative likelihood for cyclic graphs
        if not nx.is_directed_acyclic_graph(causal_g):
            return -1e15, np.zeros([len(self._event_names), len(self._event_names)]), np.zeros(len(self._event_names))

        # Initialize alpha matrix and mu vector
        alpha = np.ones([self._max_hop + 1, len(self._event_names), len(self._event_names)]) * edge_mat
        mu = np.ones(len(self._event_names))
        l_init = 0

        # Loop through each event type to update its parameters
        for i in range(len(self._event_names)):
            pa_i = set(np.where(edge_mat[:, i] == 1)[0])
            li = -1e15  # Initialize log-likelihood for event type i

            # Filter events for the current type
            ind = np.where(self._event_indexes == i)
            x_i = self.tensor['times'].values[ind]
            x_i_all = np.zeros_like(self.tensor['times'].values)
            x_i_all[ind] = x_i

            # E-M loop for parameter refinement
            while True:
                # Calculate the intensity function (lambda) sum component for likelihood
                lambda_i_sum = (self._decay_effects * alpha[:, :, i].T).sum() + mu[i] * self._T

                # Calculate the lambda values for all tensor events for event type i
                lambda_for_i = np.zeros(len(self.tensor)) + mu[i]
                for k in range(self._max_hop + 1):
                    lambda_for_i += np.matmul(self._effect_tensor_decays[k, :], alpha[k, :, i].T)
                lambda_for_i = lambda_for_i[ind]

                # Compute the new log-likelihood
                x_log_lambda = (x_i * np.log(lambda_for_i)).sum()
                new_li = -lambda_i_sum + x_log_lambda

                # Check for convergence
                delta = new_li - li
                if delta < 0.1:
                    li = new_li
                    l_init += li
                    break

                # Update old likelihood and refine parameters
                li = new_li
                mu[i] = ((mu[i] / lambda_for_i) * x_i).sum() / self._T

                # Update alpha values for all parent nodes of event i
                for j in pa_i:
                    for k in range(self._max_hop + 1):
                        upper = ((alpha[k, j, i] * (self._effect_tensor_decays[k, :, j])[ind] / lambda_for_i) * x_i).sum()
                        lower = self._decay_effects[j, k]
                        if lower == 0:
                            alpha[k, j, i] = 0
                            continue
                        alpha[k, j, i] = upper / lower

        # Apply model selection penalty
        if self._penalty == 'AIC':
            return l_init - (len(self._event_names) + self._epsilon * edge_mat.sum() * (self._max_hop + 1)), alpha, mu
        elif self._penalty == 'BIC':
            return l_init - (len(self._event_names) + self._epsilon * edge_mat.sum() * (self._max_hop + 1)) * np.log(self.tensor['times'].sum()) / 2, alpha, mu
        else:
            raise ValueError("The penalty's value should be BIC or AIC.")


    def _one_step_change_iterator(self, edge_mat):
        """
        Generate adjacency matrices that differ from the input matrix by one edge.

        This function is an iterator that yields adjacency matrices. Each yielded matrix is a one-step modification
        (addition, deletion, or reversal of an edge) of the input adjacency matrix.

        Parameters
        ----------
        edge_mat : np.ndarray
            The adjacency matrix that serves as the basis for generating new matrices.

        Returns
        -------
        iterator : iterator of np.ndarray
            An iterator that yields adjacency matrices with one edge changed compared to `edge_mat`.

        Attributes Modified
        -------------------
        None

        Notes
        -----
        Assumes that `_event_names` attribute is already initialized.

        Examples
        --------
        >>> for new_edge_mat in _one_step_change_iterator(edge_mat):
        >>>     # Perform some operations on new_edge_mat
        """

        # The function uses Python's map to apply _one_step_change function on every pair of event indices.
        # The `product(range(len(self._event_names)), range(len(self._event_names)))` generates all possible pairs
        # of indices in the adjacency matrix, allowing us to explore every possible one-step change.
        return map(lambda e: self._one_step_change(edge_mat, e),
                   product(range(len(self._event_names)),
                           range(len(self._event_names))))


    @staticmethod
    def _one_step_change(edge_mat, e):
        """
        Create a modified adjacency matrix based on a one-step edge change.

        Given an adjacency matrix and a pair of node indices (j, i), this method toggles the value of the edge between
        node j and node i. If the edge exists, it's removed; if it doesn't exist, it's added.

        Parameters
        ----------
        edge_mat : np.ndarray
            The original adjacency matrix.
        e : tuple of int
            A tuple containing the indices (j, i) that specify the edge to be toggled in the adjacency matrix.

        Returns
        -------
        new_edge_mat : np.ndarray
            The modified adjacency matrix after toggling the edge specified by (j, i).

        Examples
        --------
        >>> _one_step_change(np.array([[0, 1], [0, 0]]), (0, 1))
        array([[0, 0],
               [0, 0]])

        >>> _one_step_change(np.array([[0, 0], [0, 0]]), (0, 1))
        array([[0, 1],
               [0, 0]])

        Notes
        -----
        - Self-loops (edges from a node to itself) are not toggled; the original matrix is returned in such cases.
        - The function assumes the input adjacency matrix is for a directed graph.
        """

        j, i = e  # unpack the tuple into source node (j) and target node (i)

        # If the source and target node are the same, return the original matrix (no self-loops allowed)
        if j == i:
            return edge_mat

        new_edge_mat = edge_mat.copy()  # Create a copy of the original adjacency matrix

        # Check the current status of the edge from j to i in the adjacency matrix.
        # If the edge exists (value = 1), remove it (set to 0).
        # Otherwise, add the edge (set to 1) and make sure the reverse edge (from i to j) is removed.
        if new_edge_mat[j, i] == 1:
            new_edge_mat[j, i] = 0
        else:
            new_edge_mat[j, i] = 1
            new_edge_mat[i, j] = 0  # Ensure the reverse edge is removed when adding a new edge

        return new_edge_mat  # Return the modified adjacency matrix

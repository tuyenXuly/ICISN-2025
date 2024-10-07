import numpy as np
import networkx as nx

def check_connectivity(positions, communication_range):
    dist_matrix = np.sqrt(np.sum((positions[:, np.newaxis] - positions) ** 2, axis=2))
    adjacency_matrix = (dist_matrix <= communication_range).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)
    graph = nx.Graph(adjacency_matrix)
    return nx.is_connected(graph)
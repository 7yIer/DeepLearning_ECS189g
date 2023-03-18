import numpy as np
import networkx as nx

def load_dataset(node_file, link_file):
    # define label mapping
    label_map = {
        'Neural_Networks': 0,
        'Probabilistic_Methods': 1,
        'Genetic_Algorithms': 2,
        'Theory': 3,
        'Case_Based': 4,
        'Reinforcement_Learning': 5,
        'Rule_Learning': 6,
        'Theory': 7
    }
    # read in node features
    with open(node_file, 'r') as f:
        nodes = f.readlines()
    # create node feature matrix
    num_nodes = len(nodes)
    num_features = len(nodes[0].split()) - 2
    node_features = np.zeros((num_nodes, num_features))
    node_labels = np.zeros(num_nodes, dtype=int)
    for i, node in enumerate(nodes):
        parts = node.split()
        node_features[i,:] = np.array([float(x) for x in parts[1:-1]])
        node_label_str = parts[-1]
        if node_label_str in label_map:
            node_labels[i] = label_map[node_label_str]
        else:
            raise ValueError(f"Unknown label: {node_label_str}")
    # read in links
    with open(link_file, 'r') as f:
        links = f.readlines()
    # create graph object
    G = nx.DiGraph()
    for link in links:
        parts = link.split()
        source = int(parts[1])
        target = int(parts[0])
        G.add_edge(source, target)
    # add node features and labels to graph object
    for i in range(num_nodes):
        G.nodes[i]['features'] = node_features[i,:]
        G.nodes[i]['label'] = node_labels[i]
    return G

dataset = load_dataset('/Users/paimannejrabi/Desktop/dd/DeepLearning_ECS189g/data/stage_5_data/cora/node','/Users/paimannejrabi/Desktop/dd/DeepLearning_ECS189g/data/stage_5_data/cora/link')
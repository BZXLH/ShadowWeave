import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
import pandas as pd
import numpy as np
import yaml

# Read configuration file
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Define HAN model
class HAN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata, num_node_types):
        super().__init__()
        self.type_embedding = torch.nn.Embedding(num_node_types, hidden_channels)
        self.han_conv = HANConv(in_channels + hidden_channels, hidden_channels, heads=8, dropout=0.6, metadata=metadata)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, type_dict):
        node_types = type_dict['node']
        type_emb = self.type_embedding(node_types)
        x = x_dict['node']
        x = torch.cat([x, type_emb], dim=1)
        x = {'node': x}
        x = self.han_conv(x, edge_index_dict)
        x = F.relu(x['node'])
        x = self.lin(x)
        return x

# Load node and edge data
nodes_df = pd.read_csv(config['data']['nodes_csv_path'])
edges_df = pd.read_csv(config['data']['edges_csv_path'])

# Parse node data
node_features = torch.tensor(nodes_df.iloc[:, 2:].values, dtype=torch.float)
node_types = torch.tensor(nodes_df['node_type'].values, dtype=torch.long)

# Encode node_type
unique_node_types = torch.unique(node_types)
node_type_encoder = {node_type.item(): i for i, node_type in enumerate(unique_node_types)}
encoded_node_types = torch.tensor([node_type_encoder[node_type.item()] for node_type in node_types], dtype=torch.long)

# Parse relationship data and encode edge types
edge_type_encoder = {}
edge_type_decoder = {}
edge_type_count = 0
edge_index_dict = {}

for _, row in edges_df.iterrows():
    src = int(row['src'])
    dst = int(row['dst'])
    edge_type = row['edge_type']
    if edge_type not in edge_type_encoder:
        edge_type_encoder[edge_type] = edge_type_count
        edge_type_decoder[edge_type_count] = edge_type
        edge_type_count += 1
    if edge_type not in edge_index_dict:
        edge_index_dict[edge_type] = [[], []]
    edge_index_dict[edge_type][0].append(src)
    edge_index_dict[edge_type][1].append(dst)

# Convert edge indices to torch.Tensor
for edge_type in edge_index_dict:
    edge_index_dict[edge_type] = torch.tensor(edge_index_dict[edge_type], dtype=torch.long)

# Initialize model parameters
in_channels = node_features.size(1)
hidden_channels = config['model']['hidden_channels']
out_channels = len(edge_type_encoder)
metadata = (['node'], [('node', edge_type, 'node') for edge_type in edge_type_encoder.keys()])
num_node_types = len(unique_node_types)

# Initialize model
model = HAN(in_channels, hidden_channels, out_channels, metadata, num_node_types)

# Load trained model weights
model_path = config['model']['model_path']
state_dict = torch.load(model_path)
# Filter out mismatched keys
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.eval()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
node_features = node_features.to(device)
encoded_node_types = encoded_node_types.to(device)
for edge_type in edge_index_dict:
    edge_index_dict[edge_type] = edge_index_dict[edge_type].to(device)

# Create heterogeneous graph data object
data = HeteroData()
data['node'].x = node_features
data['node'].type = encoded_node_types
for edge_type in edge_index_dict:
    data[('node', edge_type, 'node')].edge_index = edge_index_dict[edge_type]
data = data.to(device)

# Prediction function
def predict_relationship(node_id1, node_id2):
    """
    Predict the relationship type between two nodes.

    Args:
        node_id1 (int): ID of the first node.
        node_id2 (int): ID of the second node.

    Returns:
        list: A list of relationship types and their probabilities sorted by probability.
    """
    num_nodes = data['node'].x.size(0)
    if node_id1 < 0 or node_id1 >= num_nodes or node_id2 < 0 or node_id2 >= num_nodes:
        raise ValueError(f"Node IDs must be between 0 and {num_nodes - 1}.")

    with torch.no_grad():
        node_embeddings = model(data.x_dict, data.edge_index_dict, {'node': data['node'].type})
        src_emb = node_embeddings[node_id1].unsqueeze(0)
        dst_emb = node_embeddings[node_id2].unsqueeze(0)
        edge_emb = src_emb + dst_emb
        relation_scores = edge_emb

        # Calculate probabilities for each relationship
        relation_probabilities = F.softmax(relation_scores, dim=1).cpu().detach().numpy()[0]

        # Sort by probability
        sorted_indices = relation_probabilities.argsort()[::-1]
        sorted_relations = [(edge_type_decoder[i], relation_probabilities[i]) for i in sorted_indices]

    return sorted_relations

# Read test data
# ==============================
# 6. Read test data and perform predictions
# ==============================

# List of test data filenames
test_files = [
    r'E:\工程\CM-GCN\Dataset\test(Security Task)\APT group and malware attribution.csv',
    r'E:\工程\CM-GCN\Dataset\test(Security Task)\APT group profiling.csv',
    r'E:\工程\CM-GCN\Dataset\test(Security Task)\Attack target prediction.csv',
    r'E:\工程\CM-GCN\Dataset\test(Security Task)\Attack technique simulation.csv',
    r'E:\工程\CM-GCN\Dataset\test(Security Task)\Malware propagation analysis.csv',
]

for file_name in test_files:
    file_path = f'{file_name}'
    try:
        # Read CSV file
        test_df = pd.read_csv(file_path, header=None)

        # Get node pair IDs from the first and second columns, and correct relationship types from the third column
        node1 = test_df[0].values
        node2 = test_df[1].values
        true_relation_types = test_df[2].values

        # Initialize counts for correct predictions and total predictions
        correct_predictions = 0
        total_predictions = len(node1)

        for n1, n2, true_relation in zip(node1, node2, true_relation_types):
            result = predict_relationship(n1, n2)

            # Get the relationship type with the highest probability
            max_relation, _ = max(result, key=lambda x: x[1])

            # Determine if the prediction is correct
            if max_relation == true_relation:
                correct_predictions += 1

        # Calculate prediction accuracy
        accuracy = correct_predictions / total_predictions

        print(f"File: {file_name} - Prediction Accuracy: {accuracy:.4f}")
    except FileNotFoundError:
        print(f"File {file_name} not found. Skipping...")
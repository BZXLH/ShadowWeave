import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import yaml

# Read configuration file
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Dynamically set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config['training']['device'] = device.type

# ==============================
# 2. Model definition (same as during training)
# ==============================

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_node_types, type_embedding_dim, dropout_rate):
        super().__init__()
        # Node type embedding layer
        self.node_type_embedding = torch.nn.Embedding(num_node_types, type_embedding_dim)

        # Use only 1 layer of heterogeneous GraphSAGE convolution
        self.conv1 = HeteroConv({
            ('node', 'to', 'node'): SAGEConv(in_channels + type_embedding_dim, out_channels)  # Directly output final dimension
        }, aggr='lstm')
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Relation prediction layer
        self.relation_pred = Linear(2 * out_channels, len(edge_type_names))

    def forward(self, x_dict, node_type_ids_dict, edge_index_dict):
        # Concatenate node features with type embeddings
        type_emb_dict = {key: self.node_type_embedding(node_type_ids_dict[key]) for key in node_type_ids_dict}
        x_dict = {key: torch.cat([x_dict[key], type_emb_dict[key]], dim=1) for key in x_dict}

        # Only 1 convolution layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.dropout(F.relu(x)) for key, x in x_dict.items()}

        return x_dict


# ==============================
# 2. Load data
# ==============================

# Path settings (please adjust according to actual situation)
nodes_df = pd.read_csv(config['data']['nodes_csv_path'])
edges_df = pd.read_csv(config['data']['edges_csv_path'])

# Process node features and types
node_features = torch.tensor(nodes_df.iloc[:, 2:].values, dtype=torch.float)
node_types = torch.tensor(nodes_df['node_type'].values, dtype=torch.long)

# Add edges and encode edge types
le = LabelEncoder()
edges_df['edge_type'] = le.fit_transform(edges_df['edge_type'])
edge_type_names = le.classes_

edge_indices = torch.tensor(edges_df[['src', 'dst']].values.T, dtype=torch.long)
edge_types = torch.tensor(edges_df['edge_type'].values, dtype=torch.long)

# Get actual in_channels
in_channels = node_features.size(1)

# Calculate the number of node types
num_node_types = node_types.max().item() + 1

# ==============================
# 3. Initialize model and load weights
# ==============================

# Initialize model
model = HeteroGraphSAGE(
    in_channels,
    config['model']['hidden_channels'],
    config['model']['out_channels'],
    num_node_types,
    config['model']['type_embedding_dim'],
    config['model']['dropout_rate']
)

# Update relation prediction layer according to the number of edge types
model.relation_pred = Linear(2 * config['model']['out_channels'], len(edge_type_names))

# Load trained model weights
model_path = config['model']['model_path']
model.load_state_dict(torch.load(model_path))
model.eval()

# Device configuration
model = model.to(device)
node_features = node_features.to(device)
node_types = node_types.to(device)
edge_indices = edge_indices.to(device)

# ==============================
# 4. Generate node embeddings
# ==============================

with torch.no_grad():
    # Create HeteroData object
    hetero_data = HeteroData()
    hetero_data['node'].x = node_features
    hetero_data['node'].y = node_types
    hetero_data['node', 'to', 'node'].edge_index = edge_indices
    hetero_data = hetero_data.to(device)

    # Get node type IDs
    node_type_ids = hetero_data['node'].y  # [num_nodes]
    node_type_ids_dict = {'node': node_type_ids}
    x_dict = {'node': hetero_data['node'].x}
    edge_index_dict = {('node', 'to', 'node'): edge_indices}

    # Forward propagation to get node embeddings
    x_dict_output = model(x_dict, node_type_ids_dict, edge_index_dict)
    embeddings = x_dict_output['node']

# ==============================
# 5. Define prediction function
# ==============================

def predict_relation(node_id_1, node_id_2):
    """
    Predict the relationship type between two nodes.

    Args:
        node_id_1 (int): ID of the first node.
        node_id_2 (int): ID of the second node.

    Returns:
        str: Predicted relationship type name.
    """
    # Check if node IDs are valid
    num_nodes = embeddings.size(0)
    if node_id_1 < 0 or node_id_1 >= num_nodes or node_id_2 < 0 or node_id_2 >= num_nodes:
        raise ValueError(f"Node IDs must be between 0 and {num_nodes - 1}.")

    # Get embeddings of the two nodes
    emb_1 = embeddings[node_id_1]
    emb_2 = embeddings[node_id_2]

    # Concatenate the two node embeddings
    combined_emb = torch.cat([emb_1, emb_2], dim=0)  # [2 * out_channels]

    # Pass the embedding to the relation prediction layer
    relation_scores = model.relation_pred(combined_emb.unsqueeze(0))  # Add batch dimension

    # Calculate probabilities for each relation
    relation_probabilities = F.softmax(relation_scores, dim=1).cpu().detach().numpy()[0]  # [num_relations]

    # Sort by probability
    sorted_indices = relation_probabilities.argsort()[::-1]
    sorted_relations = [(edge_type_names[i], relation_probabilities[i]) for i in sorted_indices]

    return sorted_relations

# ==============================
# 6. Read test data and make predictions
# ==============================

# Read CSV file
test_df = pd.read_csv(config['data']['test_csv_path'], header=None)

# Assume the first and second columns are the IDs of node pairs
node1 = test_df[0].values  # First column data
node2 = test_df[1].values  # Second column data

# Initialize counters for correct predictions and total predictions for different cases
correct_uses = 0
correct_indicates = 0
correct_uses_or_indicates = 0
total_predictions = len(node1)

for n1, n2 in zip(node1, node2):
    result = predict_relation(n1, n2)
    print(f"Relation prediction between node {n1} and node {n2}:")
    print(result)

    # Get the relationship type with the highest probability
    max_relation, max_prob = max(result, key=lambda x: x[1])

    # Judge different cases respectively
    if max_relation == 'uses':
        correct_uses += 1
        correct_uses_or_indicates += 1
    elif max_relation == 'indicates':
        correct_indicates += 1
        correct_uses_or_indicates += 1

# Calculate prediction accuracy for different cases
accuracy_uses = correct_uses / total_predictions
accuracy_indicates = correct_indicates / total_predictions
accuracy_uses_or_indicates = correct_uses_or_indicates / total_predictions

print(f"Prediction Accuracy for 'uses': {accuracy_uses:.4f}")
print(f"Prediction Accuracy for 'indicates': {accuracy_indicates:.4f}")
print(f"Prediction Accuracy for 'uses' or 'indicates': {accuracy_uses_or_indicates:.4f}")
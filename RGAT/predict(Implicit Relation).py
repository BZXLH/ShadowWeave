import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGATConv, Linear
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import yaml

# Read configuration file
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


# ==============================
# 2. Model definition (same as during training)
# ==============================

class HeteroRGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_node_types, type_embedding_dim, dropout_rate, heads):
        super().__init__()
        self.node_type_embedding = torch.nn.Embedding(num_node_types, type_embedding_dim)

        # 1-layer RGAT convolution: output dimension is out_channels × heads (multi-head concatenation)
        self.conv1 = RGATConv(
            in_channels + type_embedding_dim,
            out_channels,  # Output dimension per head
            num_relations,
            heads=heads  # Number of attention heads
        )
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Relation prediction layer: input dimension = 2 × (out_channels × heads) (concatenated features of two nodes)
        self.relation_pred = Linear(2 * out_channels * heads, num_relations)  # Key modification

    def forward(self, x, node_type_ids, edge_index, edge_type):
        type_emb = self.node_type_embedding(node_type_ids)
        x = torch.cat([x, type_emb], dim=1)  # Concatenate node features and type embeddings
        x = self.conv1(x, edge_index, edge_type)  # Output dimension: out_channels × heads
        x = self.dropout(F.relu(x))
        return x

# ==============================
# 2. Load data
# ==============================

# Path settings (read from configuration file)
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

# ==============================
# 3. Initialize model and load weights
# ==============================

# Parameter settings (must be consistent with training)
num_node_types = node_types.max().item() + 1
type_embedding_dim = config['model']['type_embedding_dim']  # Embedding dimension read from configuration file
in_channels = node_features.size(1)
hidden_channels = config['model']['hidden_channels']
out_channels = config['model']['out_channels']
num_relations = len(edge_type_names)
heads=config['model'].get('heads', 8)
# Initialize model
model = HeteroRGAT(in_channels, hidden_channels, out_channels, num_relations, num_node_types, type_embedding_dim, config['model']['dropout_rate'],heads)

# Load trained model weights
model_path = config['model']['model_path']
model.load_state_dict(torch.load(model_path))
model.eval()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
node_features = node_features.to(device)
node_types = node_types.to(device)
edge_indices = edge_indices.to(device)
edge_types = edge_types.to(device)

# ==============================
# 4. Generate node embeddings
# ==============================

with torch.no_grad():
    # Create HeteroData object
    hetero_data = HeteroData()
    hetero_data['node'].x = node_features
    hetero_data['node'].y = node_types
    hetero_data['node', 'to', 'node'].edge_index = edge_indices
    hetero_data['node', 'to', 'node'].edge_type = edge_types
    hetero_data = hetero_data.to(device)

    # Get node type IDs
    node_type_ids = hetero_data['node'].y  # [num_nodes]

    # Forward propagation to get node embeddings
    embeddings = model(hetero_data['node'].x, node_type_ids, hetero_data['node', 'to', 'node'].edge_index, hetero_data['node', 'to', 'node'].edge_type)

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

    # Pass the embeddings to the relation prediction layer
    relation_scores = model.relation_pred(combined_emb.unsqueeze(0))  # Add batch dimension

    # Calculate probabilities for each relation
    relation_probabilities = F.softmax(relation_scores, dim=1).cpu().detach().numpy()[0]  # [num_relations]
    # print(relation_probabilities)

    # Sort by probability
    sorted_indices = relation_probabilities.argsort()[::-1]
    sorted_relations = [(edge_type_names[i], relation_probabilities[i]) for i in sorted_indices]

    return sorted_relations


# ==============================
# 6. User interaction (optional)
# ==============================
# relation = predict_relation(1, 96)
# print(relation)
# Read CSV file
test_df = pd.read_csv(config['data']['test_csv_path'], header=None)

# Assume the first and second columns are IDs of node pairs
node1 = test_df[0].values  # Data from first column
node2 = test_df[1].values  # Data from second column

# Initialize counters for correct predictions and total predictions in different cases
correct_uses = 0
correct_indicates = 0
correct_uses_or_indicates = 0
total_predictions = len(node1)

for n1, n2 in zip(node1, node2):
    result = predict_relation(n1, n2)
    print(f"Relation prediction between node {n1} and node {n2}:")
    print(result)

    # Get the relation type with maximum probability
    max_relation, max_prob = max(result, key=lambda x: x[1])

    # Judge different cases separately
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
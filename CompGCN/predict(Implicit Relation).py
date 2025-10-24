import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import yaml

# Read configuration file
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)


# ==============================
# Modification 1: CompGCN model definition (must be consistent with training)
# ==============================
class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, act=lambda x: x, dropout=0.0):
        super().__init__(aggr='mean')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.act = act
        self.dropout = dropout

        self.rel_emb = torch.nn.Embedding(num_relations, in_channels)
        self.w_loop = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.w_in = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.w_out = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))

        torch.nn.init.xavier_uniform_(self.w_loop)
        torch.nn.init.xavier_uniform_(self.w_in)
        torch.nn.init.xavier_uniform_(self.w_out)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        edge_index, edge_type = add_self_loops(edge_index, edge_type, num_nodes=x.size(0))
        rel_emb = self.rel_emb(edge_type)
        out = self.propagate(edge_index, x=x, rel_emb=rel_emb)
        out = self.act(out)
        out = self.dropout(out)
        return out

    def message(self, x_j, rel_emb):
        return x_j * rel_emb

    def update(self, aggr_out, x):
        return aggr_out @ self.w_out + x @ self.w_loop


class HeteroCompGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_node_types, type_embedding_dim,
                 dropout_rate):
        super().__init__()
        self.node_type_embedding = torch.nn.Embedding(num_node_types, type_embedding_dim)

        self.conv1 = CompGCNConv(in_channels + type_embedding_dim, hidden_channels, num_relations,
                                 act=F.relu, dropout=dropout_rate)
        self.conv2 = CompGCNConv(hidden_channels, hidden_channels, num_relations,
                                 act=F.relu, dropout=dropout_rate)
        self.conv3 = CompGCNConv(hidden_channels, out_channels, num_relations,
                                 act=lambda x: x, dropout=dropout_rate)

        self.relation_pred = torch.nn.Linear(2 * out_channels, num_relations)

    def forward(self, x, node_type_ids, edge_index, edge_type):
        type_emb = self.node_type_embedding(node_type_ids)
        x = torch.cat([x, type_emb], dim=1)
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        x = self.conv3(x, edge_index, edge_type)
        return x


# ==============================
# 2. Load data (keep original logic)
# ==============================
nodes_df = pd.read_csv(config['data']['nodes_csv_path'])
edges_df = pd.read_csv(config['data']['edges_csv_path'])

node_features = torch.tensor(nodes_df.iloc[:, 2:].values, dtype=torch.float)
node_types = torch.tensor(nodes_df['node_type'].values, dtype=torch.long)

le = LabelEncoder()
edges_df['edge_type'] = le.fit_transform(edges_df['edge_type'])
edge_type_names = le.classes_

edge_indices = torch.tensor(edges_df[['src', 'dst']].values.T, dtype=torch.long)
edge_types = torch.tensor(edges_df['edge_type'].values, dtype=torch.long)

# ==============================
# 3. Initialize CompGCN model and load weights
# ==============================
num_node_types = node_types.max().item() + 1
type_embedding_dim = config['model']['type_embedding_dim']
in_channels = node_features.size(1)
hidden_channels = config['model']['hidden_channels']
out_channels = config['model']['out_channels']
num_relations = len(edge_type_names)

# Use CompGCN model
model = HeteroCompGCN(in_channels, hidden_channels, out_channels, num_relations,
                      num_node_types, type_embedding_dim, config['model']['dropout_rate'])

# Load weights
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
# 4. Generate node embeddings (logic unchanged)
# ==============================
with torch.no_grad():
    hetero_data = HeteroData()
    hetero_data['node'].x = node_features
    hetero_data['node'].y = node_types
    hetero_data['node', 'to', 'node'].edge_index = edge_indices
    hetero_data['node', 'to', 'node'].edge_type = edge_types
    hetero_data = hetero_data.to(device)

    node_type_ids = hetero_data['node'].y
    embeddings = model(hetero_data['node'].x, node_type_ids,
                       hetero_data['node', 'to', 'node'].edge_index,
                       hetero_data['node', 'to', 'node'].edge_type)


# ==============================
# 5. Prediction function (keep original logic)
# ==============================
def predict_relation(node_id_1, node_id_2):
    num_nodes = embeddings.size(0)
    if node_id_1 < 0 or node_id_1 >= num_nodes or node_id_2 < 0 or node_id_2 >= num_nodes:
        raise ValueError(f"Node IDs must be between 0 and {num_nodes - 1}.")

    emb_1 = embeddings[node_id_1]
    emb_2 = embeddings[node_id_2]

    combined_emb = torch.cat([emb_1, emb_2], dim=0)
    relation_scores = model.relation_pred(combined_emb.unsqueeze(0))
    relation_probabilities = F.softmax(relation_scores, dim=1).cpu().detach().numpy()[0]

    sorted_indices = relation_probabilities.argsort()[::-1]
    sorted_relations = [(edge_type_names[i], relation_probabilities[i]) for i in sorted_indices]

    return sorted_relations


# ==============================
# 6. Test logic (keep original logic)
# ==============================
# Read test data
test_df = pd.read_csv(config['data']['test_csv_path'], header=None)
node1 = test_df[0].values
node2 = test_df[1].values

# Initialize counters for correct predictions and total predictions for different cases
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
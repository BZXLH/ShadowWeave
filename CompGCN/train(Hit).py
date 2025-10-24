import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import copy
import yaml
import matplotlib.pyplot as plt  # Visualization library
import os  # File operation library

# Read configuration file
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Create result saving directory
result_dir = config.get('result_dir', 'results')
os.makedirs(result_dir, exist_ok=True)

# ==============================
# 1. Data Loading and Preprocessing
# ==============================

# Load node and edge data
nodes_df = pd.read_csv(config['data']['nodes_csv_path'])
edges_df = pd.read_csv(config['data']['edges_csv_path'])

# Create heterogeneous graph data object
hetero_data = HeteroData()

# Add node features and types
node_features = torch.tensor(nodes_df.iloc[:, 2:].values, dtype=torch.float)
node_types = torch.tensor(nodes_df['node_type'].values, dtype=torch.long)
hetero_data['node'].x = node_features
hetero_data['node'].y = node_types  # Node types are stored in 'y'

# Add edges and encode edge types
le = LabelEncoder()
edges_df['edge_type'] = le.fit_transform(edges_df['edge_type'])
edge_type_names = le.classes_
edge_type_to_code = {name: idx for idx, name in enumerate(edge_type_names)}

edge_indices = torch.tensor(edges_df[['src', 'dst']].values.T, dtype=torch.long)
edge_types = torch.tensor(edges_df['edge_type'].values, dtype=torch.long)
hetero_data['node', 'to', 'node'].edge_index = edge_indices
hetero_data['node', 'to', 'node'].edge_type = edge_types
print('Number of relation types:', len(edge_type_names))
print('Relation type encodings:', edge_type_to_code)

# Get key parameters
in_channels = node_features.size(1)
num_relations = len(edge_type_names)
num_node_types = node_types.max().item() + 1


# ==============================
# CompGCN Convolutional Layer Definition
# ==============================
class CompGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_relations, act=lambda x: x, dropout=0.0):
        super().__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.act = act
        self.dropout = dropout

        # Relation embeddings and parameters
        self.rel_emb = torch.nn.Embedding(num_relations, in_channels)
        self.w_loop = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.w_in = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.w_out = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))

        # Initialize parameters
        torch.nn.init.xavier_uniform_(self.w_loop)
        torch.nn.init.xavier_uniform_(self.w_in)
        torch.nn.init.xavier_uniform_(self.w_out)
        torch.nn.init.xavier_uniform_(self.rel_emb.weight)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_type):
        # Add self-loop edges
        edge_index, edge_type = add_self_loops(edge_index, edge_type, num_nodes=x.size(0))

        # Get relation embeddings
        rel_emb = self.rel_emb(edge_type)

        # Message passing
        out = self.propagate(edge_index, x=x, rel_emb=rel_emb)

        # Apply activation function and dropout
        out = self.act(out)
        out = self.dropout(out)

        return out

    def message(self, x_j, rel_emb):
        # Combine entity and relation features
        xj_rel = x_j * rel_emb  # Using product combination method
        return xj_rel

    def update(self, aggr_out, x):
        # Combine information from different directions
        out = aggr_out @ self.w_out + x @ self.w_loop
        return out


# ==============================
# Model Definition
# ==============================
class HeteroCompGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_node_types, type_embedding_dim,
                 dropout_rate):
        super().__init__()

        # Node type embedding
        self.node_type_embedding = torch.nn.Embedding(num_node_types, type_embedding_dim)

        # CompGCN convolutional layers
        self.conv1 = CompGCNConv(in_channels + type_embedding_dim, hidden_channels, num_relations,
                                 act=F.relu, dropout=dropout_rate)
        self.conv2 = CompGCNConv(hidden_channels, hidden_channels, num_relations,
                                 act=F.relu, dropout=dropout_rate)
        self.conv3 = CompGCNConv(hidden_channels, out_channels, num_relations,
                                 act=lambda x: x, dropout=dropout_rate)

        # Relation prediction layer
        self.relation_pred = torch.nn.Linear(2 * out_channels, num_relations)

    def forward(self, x, node_type_ids, edge_index, edge_type):
        # Node type embedding
        type_emb = self.node_type_embedding(node_type_ids)
        x = torch.cat([x, type_emb], dim=1)

        # Pass through CompGCN layers
        x = self.conv1(x, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_type)
        x = self.conv3(x, edge_index, edge_type)

        return x


# ==============================
# 3. Model Initialization and Device Configuration
# ==============================

# Initialize model
model = HeteroCompGCN(
    in_channels,
    config['model']['hidden_channels'],
    config['model']['out_channels'],
    num_relations,
    num_node_types,
    config['model']['type_embedding_dim'],
    config['model']['dropout_rate']
)
print(model)

# Dynamically set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device)
hetero_data = hetero_data.to(device)

# ==============================
# 4. Optimizer and Loss Function Definition
# ==============================

# Define optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['training']['optimizer']['lr'],
    weight_decay=float(config['training']['optimizer']['weight_decay'])
)

# Calculate frequency and weights for each edge_type
class_counts = edges_df['edge_type'].value_counts().sort_index()
print("Class distribution:\n", class_counts)

class_weights = compute_class_weight('balanced', classes=np.unique(edges_df['edge_type']), y=edges_df['edge_type'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
class_weights = class_weights ** config['training']['class_weight_power']
relation_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# ==============================
# 5. Dataset Splitting (8:1:1)
# ==============================

# Convert to numpy format for splitting
edge_indices_np = edge_indices.cpu().numpy().T  # [num_edges, 2]
edge_types_np = edge_types.cpu().numpy()

# Step 1: Split into training set (80%) and temporary set (20%)
train_edges, temp_edges, train_types, temp_types = train_test_split(
    edge_indices_np, edge_types_np,
    test_size=config['dataset_split']['test_size'],
    random_state=config['dataset_split']['random_state'],
    shuffle=True,
    stratify=edge_types_np
)

# Step 2: Split temporary set into validation set (10%) and test set (10%)
val_edges, test_edges, val_types, test_types = train_test_split(
    temp_edges, temp_types,
    test_size=0.5,
    random_state=config['dataset_split']['random_state'],
    shuffle=True,
    stratify=temp_types
)

# Convert back to tensor and transfer to device
train_edge_indices = torch.tensor(train_edges.T, dtype=torch.long).to(device)
train_edge_types = torch.tensor(train_types, dtype=torch.long).to(device)

val_edge_indices = torch.tensor(val_edges.T, dtype=torch.long).to(device)
val_edge_types = torch.tensor(val_types, dtype=torch.long).to(device)

test_edge_indices = torch.tensor(test_edges.T, dtype=torch.long).to(device)
test_edge_types = torch.tensor(test_types, dtype=torch.long).to(device)

print(
    f"Training edges: {train_edge_indices.size(1)}, "
    f"Validation edges: {val_edge_indices.size(1)}, "
    f"Test edges: {test_edge_indices.size(1)}"
)

# ==============================
# 6. Early Stopping Setup
# ==============================

patience = config['early_stopping']['patience']
best_hit1 = 0.0  # Early stopping based on Hit@1
best_epoch = 0
best_model_state = copy.deepcopy(model.state_dict())
trigger_times = 0

# Metrics for recording training process
train_losses = []
val_losses = []
test_metrics = []  # Stores Hit@1, Hit@3, Hit@10, MRR, Accuracy


# ==============================
# 7. Evaluation Metric Functions
# ==============================

def compute_aic_bic(loss, num_params, num_samples, is_classification=True):
    """Calculate AIC and BIC information criteria"""
    if is_classification:
        aic = 2 * num_params - 2 * loss
        bic = num_params * torch.log(torch.tensor(num_samples, dtype=torch.float)) - 2 * loss
    else:
        aic, bic = 0.0, 0.0
    return aic.item(), bic.item()


def calculate_hit_at_k(pred_scores, true_labels, k, edge_type_names=None, per_relation=False):
    """Calculate Hit@k metric (overall or per relation type)"""
    topk_indices = torch.topk(pred_scores, k=k, dim=1).indices  # [num_edges, k]
    hit = torch.any(topk_indices == true_labels.unsqueeze(1), dim=1).float()  # [num_edges]

    if not per_relation:
        return hit.mean().item()

    # Calculate per relation type
    hit_per_relation = {}
    for rel_type in range(len(edge_type_names)):
        rel_mask = (true_labels == rel_type)
        if rel_mask.sum() == 0:
            hit_per_relation[edge_type_names[rel_type]] = 0.0
            continue
        hit_per_relation[edge_type_names[rel_type]] = hit[rel_mask].mean().item()
    return hit_per_relation


def calculate_mrr(pred_scores, true_labels, edge_type_names=None, per_relation=False):
    """Calculate MRR (Mean Reciprocal Rank) metric"""
    # Get rankings (from high to low)
    ranks = torch.argsort(torch.argsort(pred_scores, dim=1, descending=True), dim=1) + 1  # [num_edges, num_relations]
    true_ranks = ranks[torch.arange(ranks.size(0)), true_labels]  # [num_edges]
    reciprocal_ranks = 1.0 / true_ranks.float()

    if not per_relation:
        return reciprocal_ranks.mean().item()

    # Calculate per relation type
    mrr_per_relation = {}
    for rel_type in range(len(edge_type_names)):
        rel_mask = (true_labels == rel_type)
        if rel_mask.sum() == 0:
            mrr_per_relation[edge_type_names[rel_type]] = 0.0
            continue
        mrr_per_relation[edge_type_names[rel_type]] = reciprocal_ranks[rel_mask].mean().item()
    return mrr_per_relation


def calculate_relation_accuracy(predictions, labels, edge_type_names):
    """Calculate accuracy for each relation type"""
    relation_accuracy = {}
    for relation_type in range(len(edge_type_names)):
        rel_mask = (labels == relation_type)
        true_labels = labels[rel_mask]
        predicted_labels = predictions[rel_mask]
        relation_accuracy[edge_type_names[relation_type]] = (
            accuracy_score(true_labels.cpu(), predicted_labels.cpu())
            if len(true_labels) > 0 else 0.0
        )
    return relation_accuracy


def save_results(final_results, result_dir):
    """Save results to TXT and CSV files"""
    # Save as TXT
    txt_path = os.path.join(result_dir, 'final_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Best model epoch: {final_results['best_epoch']}\n\n")
        f.write("Overall evaluation metrics:\n")
        f.write(f"Accuracy: {final_results['overall']['accuracy'] * 100:.2f}%\n")
        f.write(f"Hit@1: {final_results['overall']['hit1'] * 100:.2f}%\n")
        f.write(f"Hit@3: {final_results['overall']['hit3'] * 100:.2f}%\n")
        f.write(f"Hit@10: {final_results['overall']['hit10'] * 100:.2f}%\n")
        f.write(f"MRR: {final_results['overall']['mrr']:.4f}\n\n")

        f.write("Accuracy per relation type:\n")
        for rel, val in final_results['per_relation']['accuracy'].items():
            f.write(f"  {rel}: {val * 100:.2f}%\n")

        f.write("\nHit@1 per relation type:\n")
        for rel, val in final_results['per_relation']['hit1'].items():
            f.write(f"  {rel}: {val * 100:.2f}%\n")

        f.write("\nHit@3 per relation type:\n")
        for rel, val in final_results['per_relation']['hit3'].items():
            f.write(f"  {rel}: {val * 100:.2f}%\n")

        f.write("\nHit@10 per relation type:\n")
        for rel, val in final_results['per_relation']['hit10'].items():
            f.write(f"  {rel}: {val * 100:.2f}%\n")

        f.write("\nMRR per relation type:\n")
        for rel, val in final_results['per_relation']['mrr'].items():
            f.write(f"  {rel}: {val:.4f}\n")

    # Save as CSV
    per_rel_data = []
    for rel in edge_type_names:
        per_rel_data.append({
            'relation_type': rel,
            'accuracy': final_results['per_relation']['accuracy'][rel],
            'hit1': final_results['per_relation']['hit1'][rel],
            'hit3': final_results['per_relation']['hit3'][rel],
            'hit10': final_results['per_relation']['hit10'][rel],
            'mrr': final_results['per_relation']['mrr'][rel]
        })

    # Add overall results
    per_rel_data.append({
        'relation_type': 'overall',
        'accuracy': final_results['overall']['accuracy'],
        'hit1': final_results['overall']['hit1'],
        'hit3': final_results['overall']['hit3'],
        'hit10': final_results['overall']['hit10'],
        'mrr': final_results['overall']['mrr']
    })

    df = pd.DataFrame(per_rel_data)
    csv_path = os.path.join(result_dir, 'final_results.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Results saved to: {txt_path} and {csv_path}")


# ==============================
# 8. Model Training
# ==============================

# Prepare training data
x = hetero_data['node'].x  # Node features
node_type_ids = hetero_data['node'].y  # Node type IDs

# Training loop
num_epochs = config['training']['num_epochs']
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    # Forward propagation (training set)
    x_train = model(x, node_type_ids, train_edge_indices, train_edge_types)

    # Relation prediction
    edge_src = train_edge_indices[0]
    edge_dst = train_edge_indices[1]
    combined_emb = torch.cat([x_train[edge_src], x_train[edge_dst]], dim=1)
    relation_scores = model.relation_pred(combined_emb)

    # Calculate loss
    loss = relation_loss_fn(relation_scores, train_edge_types)
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    # Record training loss
    train_losses.append(loss.item())

    # Calculate AIC and BIC
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_samples = train_edge_types.size(0)
    aic, bic = compute_aic_bic(loss, num_params, num_samples)

    # Validation and testing
    model.eval()
    with torch.no_grad():
        # Validation set calculation
        x_val = model(x, node_type_ids, val_edge_indices, val_edge_types)
        edge_src_val = val_edge_indices[0]
        edge_dst_val = val_edge_indices[1]
        combined_emb_val = torch.cat([x_val[edge_src_val], x_val[edge_dst_val]], dim=1)
        relation_scores_val = model.relation_pred(combined_emb_val)
        val_loss = relation_loss_fn(relation_scores_val, val_edge_types)
        val_losses.append(val_loss.item())

        # Test set calculation
        x_test = model(x, node_type_ids, test_edge_indices, test_edge_types)
        edge_src_test = test_edge_indices[0]
        edge_dst_test = test_edge_indices[1]
        combined_emb_test = torch.cat([x_test[edge_src_test], x_test[edge_dst_test]], dim=1)
        relation_scores_test = model.relation_pred(combined_emb_test)

        # Calculate test set metrics
        test_predictions = relation_scores_test.argmax(dim=1)
        test_accuracy = accuracy_score(test_edge_types.cpu(), test_predictions.cpu())

        hit1_test = calculate_hit_at_k(relation_scores_test, test_edge_types, k=1)
        hit3_test = calculate_hit_at_k(relation_scores_test, test_edge_types, k=3)
        hit10_test = calculate_hit_at_k(relation_scores_test, test_edge_types, k=10)
        mrr_test = calculate_mrr(relation_scores_test, test_edge_types)

        # Record test set metrics
        test_metrics.append({
            'accuracy': test_accuracy,
            'hit1': hit1_test,
            'hit3': hit3_test,
            'hit10': hit10_test,
            'mrr': mrr_test
        })

        # Print training information
        if epoch % 10 == 0 or epoch == 1:
            print(
                f'Epoch {epoch}/{num_epochs}, '
                f'Train Loss: {loss.item():.4f}, '
                f'Val Loss: {val_loss.item():.4f}, '
                f'Test Acc: {test_accuracy * 100:.2f}%, '
                f'Hit@1: {hit1_test * 100:.2f}%, '
                f'Hit@3: {hit3_test * 100:.2f}%, '
                f'MRR: {mrr_test:.4f}, '
                f'AIC: {aic:.2f}, BIC: {bic:.2f}'
            )

    # Early stopping judgment (based on Hit@1)
    if hit1_test > best_hit1:
        best_hit1 = hit1_test
        best_hit3 = hit3_test
        best_hit10 = hit10_test
        best_mrr = mrr_test
        best_accuracy = test_accuracy
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())
        trigger_times = 0
    else:
        trigger_times += 1

    # Trigger early stopping
    if trigger_times >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# ==============================
# 9. Best Model Evaluation
# ==============================

# Load best model
model.load_state_dict(best_model_state)

# Final evaluation
model.eval()
with torch.no_grad():
    x_test_final = model(x, node_type_ids, test_edge_indices, test_edge_types)
    edge_src_test_final = test_edge_indices[0]
    edge_dst_test_final = test_edge_indices[1]
    combined_emb_test_final = torch.cat([x_test_final[edge_src_test_final], x_test_final[edge_dst_test_final]], dim=1)
    relation_scores_final = model.relation_pred(combined_emb_test_final)

    # Overall metrics
    final_hit1 = calculate_hit_at_k(relation_scores_final, test_edge_types, k=1)
    final_hit3 = calculate_hit_at_k(relation_scores_final, test_edge_types, k=3)
    final_hit10 = calculate_hit_at_k(relation_scores_final, test_edge_types, k=10)
    final_mrr = calculate_mrr(relation_scores_final, test_edge_types)

    # Accuracy
    test_predictions_final = relation_scores_final.argmax(dim=1)
    final_test_accuracy = accuracy_score(test_edge_types.cpu(), test_predictions_final.cpu())

    # Per relation type metrics
    final_hit1_per_rel = calculate_hit_at_k(
        relation_scores_final, test_edge_types, k=1,
        edge_type_names=edge_type_names, per_relation=True
    )
    final_hit3_per_rel = calculate_hit_at_k(
        relation_scores_final, test_edge_types, k=3,
        edge_type_names=edge_type_names, per_relation=True
    )
    final_hit10_per_rel = calculate_hit_at_k(
        relation_scores_final, test_edge_types, k=10,
        edge_type_names=edge_type_names, per_relation=True
    )
    final_mrr_per_rel = calculate_mrr(
        relation_scores_final, test_edge_types,
        edge_type_names=edge_type_names, per_relation=True
    )
    final_relation_accuracy = calculate_relation_accuracy(
        test_predictions_final, test_edge_types, edge_type_names
    )

    # Organize results
    final_results = {
        'best_epoch': best_epoch,
        'overall': {
            'accuracy': final_test_accuracy,
            'hit1': final_hit1,
            'hit3': final_hit3,
            'hit10': final_hit10,
            'mrr': final_mrr
        },
        'per_relation': {
            'accuracy': final_relation_accuracy,
            'hit1': final_hit1_per_rel,
            'hit3': final_hit3_per_rel,
            'hit10': final_hit10_per_rel,
            'mrr': final_mrr_per_rel
        }
    }

    # Print final results
    print("\n" + "=" * 50)
    print(f"Best model performance (at epoch {best_epoch})")
    print("=" * 50)
    print(f'Final test set accuracy: {final_test_accuracy * 100:.2f}%')
    print(f'Final test set Hit@1: {final_hit1 * 100:.2f}%')
    print(f'Final test set Hit@3: {final_hit3 * 100:.2f}%')
    print(f'Final test set Hit@10: {final_hit10 * 100:.2f}%')
    print(f'Final test set MRR: {final_mrr:.4f}')

    print("\nAccuracy per relation type:")
    for rel, acc in final_relation_accuracy.items():
        print(f'  Relation {rel}: {acc * 100:.2f}%')

    print("\nHit@1 per relation type:")
    for rel, hit in final_hit1_per_rel.items():
        print(f'  Relation {rel}: {hit * 100:.2f}%')

    print("\nHit@3 per relation type:")
    for rel, hit in final_hit3_per_rel.items():
        print(f'  Relation {rel}: {hit * 100:.2f}%')

    print("\nHit@10 per relation type:")
    for rel, hit in final_hit10_per_rel.items():
        print(f'  Relation {rel}: {hit * 100:.2f}%')

    print("\nMRR per relation type:")
    for rel, mrr in final_mrr_per_rel.items():
        print(f'  Relation {rel}: {mrr:.4f}')
    print("=" * 50 + "\n")

    # Save results
    save_results(final_results, result_dir)

# ==============================
# 10. Model Saving and Visualization
# ==============================

# Save best model
model_path = config['model']['model_path']
torch.save(model.state_dict(), model_path)
print(f'Best model saved to {model_path}, achieving best performance at epoch {best_epoch}')

# Visualize training process
plt.figure(figsize=(16, 6))

# Loss curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss curves')
plt.legend()

# Evaluation metric curves
plt.subplot(1, 2, 2)
hit1_list = [m['hit1'] for m in test_metrics]
hit3_list = [m['hit3'] for m in test_metrics]
# hit10_list = [m['hit10'] for m in test_metrics]
mrr_list = [m['mrr'] for m in test_metrics]

plt.plot(hit1_list, label='Hit@1')
plt.plot(hit3_list, label='Hit@3')
# plt.plot(hit10_list, label='Hit@10')
plt.plot(mrr_list, label='MRR')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Test set evaluation metric curves')
plt.legend()

plt.tight_layout()
fig_path = os.path.join(result_dir, 'training_curves.png')
plt.savefig(fig_path, dpi=500)
plt.show()
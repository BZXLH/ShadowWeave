import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, HeteroConv, Linear
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import copy
import yaml
import matplotlib.pyplot as plt
import os  # New: used for creating save directories

# Read configuration file
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Create result save directory
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
hetero_data['node'].y = node_types  # Node types stored in 'y'

# Add edges and encode edge types
le = LabelEncoder()
edges_df['edge_type'] = le.fit_transform(edges_df['edge_type'])
edge_type_names = le.classes_
# Create mapping dictionary from edge type to code for subsequent queries
edge_type_to_code = {name: idx for idx, name in enumerate(edge_type_names)}

edge_indices = torch.tensor(edges_df[['src', 'dst']].values.T, dtype=torch.long)
edge_types = torch.tensor(edges_df['edge_type'].values, dtype=torch.long)
hetero_data['node', 'to', 'node'].edge_index = edge_indices
hetero_data['node', 'to', 'node'].edge_type = edge_types
print('Number of relation types:', len(edge_type_names))
print('Relation type encodings:', edge_type_to_code)

# Get actual input feature dimension
in_channels = node_features.size(1)

# Calculate number of node types
num_node_types = node_types.max().item() + 1


# ==============================
# 2. Model Definition
# ==============================

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_node_types, type_embedding_dim, dropout_rate):
        super().__init__()
        # Node type embedding layer
        self.node_type_embedding = torch.nn.Embedding(num_node_types, type_embedding_dim)

        # Define GraphSAGE convolution layers for heterogeneous graph
        self.conv1 = HeteroConv({
            ('node', 'to', 'node'): SAGEConv(in_channels + type_embedding_dim, hidden_channels)
        }, aggr='lstm')
        self.conv2 = HeteroConv({
            ('node', 'to', 'node'): SAGEConv(hidden_channels, hidden_channels)
        }, aggr='lstm')
        self.conv3 = HeteroConv({
            ('node', 'to', 'node'): SAGEConv(hidden_channels, out_channels)
        }, aggr='lstm')
        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # Relation prediction layer
        self.relation_pred = Linear(2 * out_channels, len(edge_type_names))

    def forward(self, x_dict, node_type_ids_dict, edge_index_dict):
        # Get node type embeddings
        type_emb_dict = {key: self.node_type_embedding(node_type_ids_dict[key]) for key in node_type_ids_dict}

        # Concatenate node features with embedding vectors
        x_dict = {key: torch.cat([x_dict[key], type_emb_dict[key]], dim=1) for key in x_dict}

        # Pass through GraphSAGE convolution layers for heterogeneous graph
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.dropout(F.relu(x)) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.dropout(F.relu(x)) for key, x in x_dict.items()}
        x_dict = self.conv3(x_dict, edge_index_dict)

        return x_dict


# ==============================
# 3. Model Initialization and Device Configuration
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
print(model)
# Dynamically set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
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

# Calculate frequency of each edge_type
class_counts = edges_df['edge_type'].value_counts().sort_index()
print("Class distribution:\n", class_counts)

# Calculate weights
# class_weights = compute_class_weight('balanced', classes=np.unique(edges_df['edge_type']), y=edges_df['edge_type'])
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)  # Convert to tensor and move to GPU
# class_weights = class_weights ** config['training']['class_weight_power']
# relation_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
relation_loss_fn = torch.nn.CrossEntropyLoss()  # Without weights
# ==============================
# 5. Dataset Splitting (8:1:1)
# ==============================

# Transpose edge_indices to fit input format of train_test_split
edge_indices_np = edge_indices.cpu().numpy().T  # Convert to shape [num_edges, 2]
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

# Convert numpy arrays back to torch tensors and move to device
train_edge_indices = torch.tensor(train_edges.T, dtype=torch.long).to(device)  # [2, num_train_edges]
train_edge_types = torch.tensor(train_types, dtype=torch.long).to(device)

val_edge_indices = torch.tensor(val_edges.T, dtype=torch.long).to(device)  # [2, num_val_edges]
val_edge_types = torch.tensor(val_types, dtype=torch.long).to(device)

test_edge_indices = torch.tensor(test_edges.T, dtype=torch.long).to(device)  # [2, num_test_edges]
test_edge_types = torch.tensor(test_types, dtype=torch.long).to(device)

print(
    f"Number of training edges: {train_edge_indices.size(1)}, Number of validation edges: {val_edge_indices.size(1)}, Number of test edges: {test_edge_indices.size(1)}")

# ==============================
# 6. Early Stopping Setup
# ==============================

# Define early stopping parameters
patience = config['early_stopping']['patience']
best_hit1 = 0.0  # Early stopping based on Hit@1
best_epoch = 0
best_model_state = copy.deepcopy(model.state_dict())  # Save state of best model

# Used for storing training and validation losses as well as test metrics
train_losses = []
val_losses = []
test_metrics = []  # Store Hit@1, Hit@3, Hit@10, MRR for each epoch


# ==============================
# 7. Auxiliary Functions
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
    """
    Calculate Hit@k metric (overall or per relation type)
    """
    # Get indices of top k predictions (sorted by score descending)
    topk_indices = torch.topk(pred_scores, k=k, dim=1).indices  # [num_edges, k]

    # Check if true labels are in top k indices (per sample)
    hit = torch.any(topk_indices == true_labels.unsqueeze(1), dim=1).float()  # [num_edges], 1 for hit, 0 for miss

    if not per_relation:
        return hit.mean().item()  # Overall Hit@k

    # Calculate Hit@k per relation type
    hit_per_relation = {}
    for rel_type in range(len(edge_type_names)):
        # Filter samples of current relation type
        rel_mask = (true_labels == rel_type)
        if rel_mask.sum() == 0:
            hit_per_relation[edge_type_names[rel_type]] = 0.0
            continue
        # Calculate Hit@k for this type of samples
        hit_rel = hit[rel_mask].mean().item()
        hit_per_relation[edge_type_names[rel_type]] = hit_rel

    return hit_per_relation


def calculate_mrr(pred_scores, true_labels, edge_type_names=None, per_relation=False):
    """
    Calculate MRR (Mean Reciprocal Rank) metric
    """
    # Sort prediction scores for each sample and get ranks (descending)
    ranks = torch.argsort(torch.argsort(pred_scores, dim=1, descending=True), dim=1) + 1  # [num_edges, num_relations]

    # Get ranks corresponding to true labels
    true_ranks = ranks[torch.arange(ranks.size(0)), true_labels]  # [num_edges]

    # Calculate reciprocal ranks
    reciprocal_ranks = 1.0 / true_ranks.float()  # [num_edges]

    if not per_relation:
        return reciprocal_ranks.mean().item()  # Overall MRR

    # Calculate MRR per relation type
    mrr_per_relation = {}
    for rel_type in range(len(edge_type_names)):
        # Filter samples of current relation type
        rel_mask = (true_labels == rel_type)
        if rel_mask.sum() == 0:
            mrr_per_relation[edge_type_names[rel_type]] = 0.0
            continue
        # Calculate MRR for this type of samples
        mrr_rel = reciprocal_ranks[rel_mask].mean().item()
        mrr_per_relation[edge_type_names[rel_type]] = mrr_rel

    return mrr_per_relation


# New: Function to save results to files
def save_results(final_results, result_dir):
    """Save final test results to TXT and CSV files"""
    # Save as TXT file
    txt_path = os.path.join(result_dir, 'final_results.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Best model epoch: {final_results['best_epoch']}\n\n")
        f.write("Overall evaluation metrics:\n")
        f.write(f"Hit@1: {final_results['overall']['hit1'] * 100:.2f}%\n")
        f.write(f"Hit@3: {final_results['overall']['hit3'] * 100:.2f}%\n")
        f.write(f"Hit@10: {final_results['overall']['hit10'] * 100:.2f}%\n")
        f.write(f"MRR: {final_results['overall']['mrr']:.4f}\n\n")

        f.write("Hit@1 per relation type:\n")
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

    # Save as CSV file (facilitates subsequent analysis)
    per_rel_data = []
    for rel in edge_type_names:
        per_rel_data.append({
            'relation_type': rel,
            'hit1': final_results['per_relation']['hit1'][rel],
            'hit3': final_results['per_relation']['hit3'][rel],
            'hit10': final_results['per_relation']['hit10'][rel],
            'mrr': final_results['per_relation']['mrr'][rel]
        })

    # Add overall result row
    per_rel_data.append({
        'relation_type': 'overall',
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

# Get node type IDs and move to device
node_type_ids = hetero_data['node'].y  # [num_nodes]
node_type_ids_dict = {'node': node_type_ids}
x_dict = {'node': hetero_data['node'].x}
edge_index_dict = {('node', 'to', 'node'): train_edge_indices}

# Train the model
num_epochs = config['training']['num_epochs']
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    # Forward propagation
    x_dict_output = model(x_dict, node_type_ids_dict, edge_index_dict)
    x = x_dict_output['node']

    # Relation prediction
    edge_src = train_edge_indices[0]
    edge_dst = train_edge_indices[1]
    combined_emb = torch.cat([x[edge_src], x[edge_dst]], dim=1)  # [num_train_edges, 2 * out_channels]
    relation_scores = model.relation_pred(combined_emb)  # [num_train_edges, num_relations]

    # Calculate loss
    loss = relation_loss_fn(relation_scores, train_edge_types)
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    # Calculate AIC and BIC
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_samples = train_edge_types.size(0)
    aic, bic = compute_aic_bic(loss, num_params, num_samples)

    # Evaluate after each epoch
    model.eval()
    with torch.no_grad():
        # Calculate for test set
        edge_index_dict_test = {('node', 'to', 'node'): test_edge_indices}
        x_dict_test = model(x_dict, node_type_ids_dict, edge_index_dict_test)
        x_test = x_dict_test['node']
        edge_src_test = test_edge_indices[0]
        edge_dst_test = test_edge_indices[1]
        combined_emb_test = torch.cat([x_test[edge_src_test], x_test[edge_dst_test]], dim=1)
        relation_scores_test = model.relation_pred(combined_emb_test)  # Keep all scores

        # Calculate for validation set
        edge_index_dict_val = {('node', 'to', 'node'): val_edge_indices}
        x_dict_val = model(x_dict, node_type_ids_dict, edge_index_dict_val)
        x_val = x_dict_val['node']
        edge_src_val = val_edge_indices[0]
        edge_dst_val = val_edge_indices[1]
        combined_emb_val = torch.cat([x_val[edge_src_val], x_val[edge_dst_val]], dim=1)
        relation_scores_val = model.relation_pred(combined_emb_val)

        # Calculate validation set loss
        val_loss = relation_loss_fn(relation_scores_val, val_edge_types)

        # Calculate test set evaluation metrics (including new MRR)
        hit1_test = calculate_hit_at_k(relation_scores_test, test_edge_types, k=1)
        hit3_test = calculate_hit_at_k(relation_scores_test, test_edge_types, k=3)
        hit10_test = calculate_hit_at_k(relation_scores_test, test_edge_types, k=10)
        mrr_test = calculate_mrr(relation_scores_test, test_edge_types)  # Calculate MRR

        # Save metrics
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        test_metrics.append({
            'hit1': hit1_test,
            'hit3': hit3_test,
            'hit10': hit10_test,
            'mrr': mrr_test  # New MRR metric
        })

        # Print training process
        if epoch % 10 == 0 or epoch == 1:
            print(
                f'Epoch {epoch}/{num_epochs}, '
                f'Train Loss: {loss.item():.4f}, '
                f'Val Loss: {val_loss.item():.4f}, '
                f'Test Hit@1: {hit1_test * 100:.2f}%, '
                f'Hit@3: {hit3_test * 100:.2f}%, '
                f'Hit@10: {hit10_test * 100:.2f}%, '
                f'MRR: {mrr_test:.4f}, '  # New MRR output
                f'AIC: {aic:.2f}, BIC: {bic:.2f}'
            )
    # Initialize early stopping counter
    trigger_times = 0
    # Early stopping judgment (based on Hit@1)
    if hit1_test > best_hit1:
        best_hit1 = hit1_test
        best_hit3 = hit3_test
        best_hit10 = hit10_test
        best_mrr = mrr_test  # Save best MRR
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())
        trigger_times = 0
    else:
        trigger_times += 1

    # Check if early stopping is triggered
    if trigger_times >= patience:
        print(f"Early stopping triggered at epoch {epoch}")
        break

# ==============================
# 9. Load Best Model and Evaluate
# ==============================

# Load best model state
model.load_state_dict(best_model_state)

# Final evaluation on test set
model.eval()
with torch.no_grad():
    edge_index_dict_test = {('node', 'to', 'node'): test_edge_indices}
    x_dict_test = model(x_dict, node_type_ids_dict, edge_index_dict_test)
    x_test = x_dict_test['node']

    edge_src_test = test_edge_indices[0]
    edge_dst_test = test_edge_indices[1]
    combined_emb_test = torch.cat([x_test[edge_src_test], x_test[edge_dst_test]], dim=1)
    relation_scores_final = model.relation_pred(combined_emb_test)

    # Overall evaluation metrics (including MRR)
    final_hit1 = calculate_hit_at_k(relation_scores_final, test_edge_types, k=1)
    final_hit3 = calculate_hit_at_k(relation_scores_final, test_edge_types, k=3)
    final_hit10 = calculate_hit_at_k(relation_scores_final, test_edge_types, k=10)
    final_mrr = calculate_mrr(relation_scores_final, test_edge_types)  # Calculate final MRR

    # Evaluation metrics per relation type (including MRR)
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
    final_mrr_per_rel = calculate_mrr(  # Calculate MRR per relation type
        relation_scores_final, test_edge_types,
        edge_type_names=edge_type_names, per_relation=True
    )

    # Organize result data structure
    final_results = {
        'best_epoch': best_epoch,
        'overall': {
            'hit1': final_hit1,
            'hit3': final_hit3,
            'hit10': final_hit10,
            'mrr': final_mrr
        },
        'per_relation': {
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
    print(f'Final test set Hit@1: {final_hit1 * 100:.2f}%')
    print(f'Final test set Hit@3: {final_hit3 * 100:.2f}%')
    print(f'Final test set Hit@10: {final_hit10 * 100:.2f}%')
    print(f'Final test set MRR: {final_mrr:.4f}')

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

    # Save final results
    save_results(final_results, result_dir)

# ==============================
# 11. Model Saving and Visualization
# ==============================

# Save best model
model_path = config['model']['model_path']
torch.save(model.state_dict(), model_path)
print(f'Best model saved to {model_path}, achieving best performance at epoch {best_epoch}')

# Visualize training process
plt.figure(figsize=(16, 6))

# Plot loss curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss curves')
plt.legend()

# Plot evaluation metric curves (including MRR)
plt.subplot(1, 2, 2)
hit1_list = [m['hit1'] for m in test_metrics]
hit3_list = [m['hit3'] for m in test_metrics]
hit10_list = [m['hit10'] for m in test_metrics]
mrr_list = [m['mrr'] for m in test_metrics]  # MRR curve data

plt.plot(hit1_list, label='Hit@1')
plt.plot(hit3_list, label='Hit@3')
plt.plot(mrr_list, label='MRR')  # Plot MRR curve
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Test set evaluation metric curves')
plt.legend()

plt.tight_layout()
fig_path = os.path.join(result_dir, 'figure.png')  # Image saved to result directory
plt.savefig(fig_path, dpi=500)  # Save as high-DPI PNG format
plt.show()
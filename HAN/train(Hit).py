import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt  # Import visualization library
import os  # New: File operation library


# Read configuration file
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Create result saving directory
result_dir = config.get('result_dir', 'results')
os.makedirs(result_dir, exist_ok=True)

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
edge_labels = []

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
    edge_labels.append(edge_type_encoder[edge_type])

edge_labels = torch.tensor(edge_labels, dtype=torch.long)
edge_type_names = [edge_type_decoder[i] for i in range(edge_type_count)]  # List of relationship type names

# Convert edge indices to torch.Tensor
for edge_type in edge_index_dict:
    edge_index_dict[edge_type] = torch.tensor(edge_index_dict[edge_type], dtype=torch.long)

# Create heterogeneous graph data object
data = HeteroData()
data['node'].x = node_features
data['node'].type = encoded_node_types
for edge_type in edge_index_dict:
    data[('node', edge_type, 'node')].edge_index = edge_index_dict[edge_type]
data['edge'].y = edge_labels


# Define HAN model
class HAN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, metadata, num_node_types):
        super().__init__()
        self.type_embedding = torch.nn.Embedding(num_node_types, hidden_channels)
        self.han_conv = HANConv(in_channels + hidden_channels, hidden_channels, heads=8, dropout=0.1, metadata=metadata)
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


# Initialize model, optimizer and loss function
in_channels = node_features.size(1)
hidden_channels = config['model']['hidden_channels']
out_channels = len(edge_type_encoder)  # Number of edge types
metadata = data.metadata()
num_node_types = len(unique_node_types)
model = HAN(in_channels, hidden_channels, out_channels, metadata, num_node_types)

# Dynamically set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

# Define optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config['training']['optimizer']['lr'],
    weight_decay=float(config['training']['optimizer']['weight_decay'])
)

# Calculate frequency and weight for each edge_type
class_counts = np.bincount(edge_labels.cpu().numpy())
print("Class Counts:\n", class_counts)

class_weights = np.bincount(edge_labels.cpu().numpy())
total = class_weights.sum()
class_weights = total / (len(class_weights) * class_weights)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
class_weights = class_weights ** config['training']['class_weight_power']
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Dataset split (8:1:1)
all_edge_indices = []
all_edge_types = []
for edge_type in edge_index_dict:
    edge_index = edge_index_dict[edge_type]
    num_edges = edge_index.size(1)
    all_edge_indices.extend([(edge_index[0][i].item(), edge_index[1][i].item()) for i in range(num_edges)])
    all_edge_types.extend([edge_type_encoder[edge_type]] * num_edges)

all_edge_indices = np.array(all_edge_indices)
all_edge_types = np.array(all_edge_types)

# Split into training set (80%), validation set (10%), test set (10%)
train_edges, temp_edges, train_types, temp_types = train_test_split(
    all_edge_indices, all_edge_types,
    test_size=config['dataset_split']['test_size'],
    random_state=config['dataset_split']['random_state'],
    shuffle=True,
    stratify=all_edge_types
)

val_edges, test_edges, val_types, test_types = train_test_split(
    temp_edges, temp_types,
    test_size=0.5,
    random_state=config['dataset_split']['random_state'],
    shuffle=True,
    stratify=temp_types
)

# Convert to tensor and move to device
train_edge_indices = torch.tensor(train_edges.T, dtype=torch.long).to(device)
train_edge_types = torch.tensor(train_types, dtype=torch.long).to(device)
val_edge_indices = torch.tensor(val_edges.T, dtype=torch.long).to(device)
val_edge_types = torch.tensor(val_types, dtype=torch.long).to(device)
test_edge_indices = torch.tensor(test_edges.T, dtype=torch.long).to(device)
test_edge_types = torch.tensor(test_types, dtype=torch.long).to(device)

print(
    f"Train edges: {train_edge_indices.size(1)}, Validation edges: {val_edge_indices.size(1)}, Test edges: {test_edge_indices.size(1)}")

# Early stopping setup
patience = config['early_stopping']['patience']
best_test_mrr = 0.0  # Early stopping based on MRR
trigger_times = 0
best_model_state = copy.deepcopy(model.state_dict())
best_epoch = 0

# Initialize training process records (for visualization)
train_losses = []  # Simplified to list storage
val_losses = []
test_metrics = {
    'MRR': [],
    'Hit@1': [],
    'Hit@3': [],
    'Hit@10': []
}


# Define evaluation metric calculation function (Hit@k and MRR)
def calculate_ranking_metrics(scores, labels, k_list=[1, 3, 10]):
    """Calculate Hit@k and MRR metrics"""
    num_samples = scores.size(0)
    num_classes = scores.size(1)

    # Sort scores in descending order and get class rankings (ranking starts at 1)
    _, sorted_indices = torch.sort(scores, dim=1, descending=True)  # [num_samples, num_classes]

    # Calculate the rank of the true label for each sample
    ranks = []
    for i in range(num_samples):
        true_label = labels[i]
        # Find the position of the true label in the sorted results (0-based index), add 1 to get the rank
        rank = (sorted_indices[i] == true_label).nonzero().item() + 1
        ranks.append(rank)
    ranks = torch.tensor(ranks, dtype=torch.float)

    # Calculate Hit@k
    hit_at_k = {f'Hit@{k}': (ranks <= k).float().mean().item() for k in k_list}

    # Calculate MRR (Mean Reciprocal Rank)
    mrr = (1.0 / ranks).mean().item()

    return {**hit_at_k, 'MRR': mrr}


def calculate_per_relation_metrics(scores, labels, edge_type_decoder):
    """Calculate Hit@k and MRR by relationship type"""
    per_relation_metrics = {}
    for rel_idx in edge_type_decoder:
        rel_type = edge_type_decoder[rel_idx]
        # Filter samples of current relationship type
        mask = (labels == rel_idx)
        if mask.sum() == 0:
            # Set metrics to 0 when there are no samples of this type
            per_relation_metrics[rel_type] = {'Hit@1': 0.0, 'Hit@3': 0.0, 'Hit@10': 0.0, 'MRR': 0.0}
            continue
        # Calculate metrics
        rel_scores = scores[mask]
        rel_labels = labels[mask]
        metrics = calculate_ranking_metrics(rel_scores, rel_labels)
        per_relation_metrics[rel_type] = metrics
    return per_relation_metrics


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

        f.write("Accuracy by relationship type:\n")
        for rel, val in final_results['per_relation']['accuracy'].items():
            f.write(f"  {rel}: {val * 100:.2f}%\n")

        f.write("\nHit@1 by relationship type:\n")
        for rel, val in final_results['per_relation']['hit1'].items():
            f.write(f"  {rel}: {val * 100:.2f}%\n")

        f.write("\nHit@3 by relationship type:\n")
        for rel, val in final_results['per_relation']['hit3'].items():
            f.write(f"  {rel}: {val * 100:.2f}%\n")

        f.write("\nHit@10 by relationship type:\n")
        for rel, val in final_results['per_relation']['hit10'].items():
            f.write(f"  {rel}: {val * 100:.2f}%\n")

        f.write("\nMRR by relationship type:\n")
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


# Train the model
num_epochs = config['training']['num_epochs']
for epoch in range(1, num_epochs + 1):
    model.train()
    optimizer.zero_grad()

    # Forward propagation
    node_embeddings = model(data.x_dict, data.edge_index_dict, {'node': data['node'].type})

    # Relationship prediction (calculate edge embeddings and scores)
    edge_src = train_edge_indices[0]
    edge_dst = train_edge_indices[1]
    src_emb = node_embeddings[edge_src]
    dst_emb = node_embeddings[edge_dst]
    edge_emb = src_emb + dst_emb  # Edge embedding calculation method
    relation_scores = edge_emb  # [num_edges, num_classes]

    # Calculate loss
    loss = criterion(relation_scores, train_edge_types)
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()  # Clear cache

    # Record training loss
    train_losses.append(loss.item())

    # Evaluate metrics on each dataset
    model.eval()
    with torch.no_grad():
        # Calculate node embeddings (reuse forward propagation result once)
        node_embeddings_all = model(data.x_dict, data.edge_index_dict, {'node': data['node'].type})

        # Validation set calculation (for loss calculation)
        src_emb_val = node_embeddings_all[val_edge_indices[0]]
        dst_emb_val = node_embeddings_all[val_edge_indices[1]]
        edge_emb_val = src_emb_val + dst_emb_val
        scores_val = edge_emb_val
        val_loss = criterion(scores_val, val_edge_types)
        val_losses.append(val_loss.item())

        # Test set metric calculation
        src_emb_test = node_embeddings_all[test_edge_indices[0]]
        dst_emb_test = node_embeddings_all[test_edge_indices[1]]
        edge_emb_test = src_emb_test + dst_emb_test
        scores_test = edge_emb_test
        metrics_test = calculate_ranking_metrics(scores_test, test_edge_types)
        for key in metrics_test:
            test_metrics[key].append(metrics_test[key])

        # Print training information (every 10 epochs or first epoch)
        if epoch % 10 == 0 or epoch == 1:
            print(
                f'Epoch {epoch}/{num_epochs} | '
                f'Train Loss: {loss.item():.4f} | '
                f'Val Loss: {val_loss.item():.4f} | '
                f'Test Hit@1: {metrics_test["Hit@1"] * 100:.2f}% | '
                f'Hit@3: {metrics_test["Hit@3"] * 100:.2f}% | '
                f'Hit@10: {metrics_test["Hit@10"] * 100:.2f}% | '
                f'MRR: {metrics_test["MRR"]:.4f}'
            )

    # Early stopping judgment (based on test set MRR)
    current_test_mrr = metrics_test["MRR"]
    if current_test_mrr > best_test_mrr:
        best_test_mrr = current_test_mrr
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())
        trigger_times = 0
    else:
        trigger_times += 1

    if trigger_times >= patience:
        print(f"Early stopping triggered at epoch {epoch}, best model at epoch {best_epoch}")
        break


# Visualize training process: combine into one figure
def plot_training_curves(train_losses, val_losses, test_metrics, save_path=None):
    """Plot loss curves and evaluation metric curves in one figure"""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    # 1. Loss curves (training + validation)
    ax1.plot(epochs, train_losses, label='Train Loss')
    ax1.plot(epochs, val_losses, label='Val Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and validation loss curves')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Evaluation metric curves (Hit@1, Hit@3, MRR)
    ax2.plot(epochs, test_metrics['Hit@1'], label='Hit@1')
    ax2.plot(epochs, test_metrics['Hit@3'], label='Hit@3')
    ax2.plot(epochs, test_metrics['MRR'], label='MRR')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('Test set evaluation metric curves')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=500)
        print(f"Training curves saved to {save_path}")
    plt.show()


# Plot and save training curves
fig_path = os.path.join(result_dir, 'training_curves.png')
plot_training_curves(train_losses, val_losses, test_metrics, save_path=fig_path)

# Load best model and perform final evaluation
model.load_state_dict(best_model_state)
model.eval()
with torch.no_grad():
    # Calculate final test set scores
    node_embeddings_final = model(data.x_dict, data.edge_index_dict, {'node': data['node'].type})
    edge_src_final = test_edge_indices[0]
    edge_dst_final = test_edge_indices[1]
    src_emb_final = node_embeddings_final[edge_src_final]
    dst_emb_final = node_embeddings_final[edge_dst_final]
    edge_emb_final = src_emb_final + dst_emb_final
    relation_scores_final = edge_emb_final

    # Calculate final test set metrics
    final_test_metrics = calculate_ranking_metrics(relation_scores_final, test_edge_types)
    final_mrr = final_test_metrics['MRR']
    final_hit1 = final_test_metrics['Hit@1']
    final_hit3 = final_test_metrics['Hit@3']
    final_hit10 = final_test_metrics['Hit@10']

    # Calculate accuracy
    test_predictions_final = relation_scores_final.argmax(dim=1)
    final_test_accuracy = accuracy_score(test_edge_types.cpu(), test_predictions_final.cpu())

    # Calculate final metrics by relationship type
    final_per_relation_metrics = calculate_per_relation_metrics(relation_scores_final, test_edge_types,
                                                                edge_type_decoder)

    # Calculate accuracy by relationship type
    final_relation_accuracy = {}
    for rel_idx in edge_type_decoder:
        rel_type = edge_type_decoder[rel_idx]
        mask = (test_edge_types == rel_idx)
        if mask.sum() == 0:
            final_relation_accuracy[rel_type] = 0.0
            continue
        true_labels = test_edge_types[mask]
        preds = test_predictions_final[mask]
        final_relation_accuracy[rel_type] = accuracy_score(true_labels.cpu(), preds.cpu())

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
            'hit1': {k: v['Hit@1'] for k, v in final_per_relation_metrics.items()},
            'hit3': {k: v['Hit@3'] for k, v in final_per_relation_metrics.items()},
            'hit10': {k: v['Hit@10'] for k, v in final_per_relation_metrics.items()},
            'mrr': {k: v['MRR'] for k, v in final_per_relation_metrics.items()}
        }
    }

    # Print final results
    print("\n" + "=" * 50)
    print(f"Best model performance (epoch {best_epoch})")
    print("=" * 50)
    print(f'Final test set accuracy: {final_test_accuracy * 100:.2f}%')
    print(f'Final test set Hit@1: {final_hit1 * 100:.2f}%')
    print(f'Final test set Hit@3: {final_hit3 * 100:.2f}%')
    print(f'Final test set Hit@10: {final_hit10 * 100:.2f}%')
    print(f'Final test set MRR: {final_mrr:.4f}')

    print("\nAccuracy by relationship type:")
    for rel, acc in final_relation_accuracy.items():
        print(f'  Relation {rel}: {acc * 100:.2f}%')

    print("\nHit@1 by relationship type:")
    for rel, metrics in final_per_relation_metrics.items():
        print(f'  Relation {rel}: {metrics["Hit@1"] * 100:.2f}%')

    print("\nHit@3 by relationship type:")
    for rel, metrics in final_per_relation_metrics.items():
        print(f'  Relation {rel}: {metrics["Hit@3"] * 100:.2f}%')

    print("\nHit@10 by relationship type:")
    for rel, metrics in final_per_relation_metrics.items():
        print(f'  Relation {rel}: {metrics["Hit@10"] * 100:.2f}%')

    print("\nMRR by relationship type:")
    for rel, metrics in final_per_relation_metrics.items():
        print(f'  Relation {rel}: {metrics["MRR"]:.4f}')
    print("=" * 50 + "\n")

    # Save results
    save_results(final_results, result_dir)

# Save best model
model_path = config['model']['model_path']
torch.save(model.state_dict(), model_path)
print(f'Best model saved to {model_path}, achieving best performance at epoch {best_epoch}')
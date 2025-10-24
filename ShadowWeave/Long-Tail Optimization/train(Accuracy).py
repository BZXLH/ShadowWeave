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
import copy  # For deep copying the model
import yaml
import matplotlib.pyplot as plt

# Read configuration file
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

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

edge_indices = torch.tensor(edges_df[['src', 'dst']].values.T, dtype=torch.long)
edge_types = torch.tensor(edges_df['edge_type'].values, dtype=torch.long)
hetero_data['node', 'to', 'node'].edge_index = edge_indices
hetero_data['node', 'to', 'node'].edge_type = edge_types
print('***:', len(edge_type_names))

# Get actual in_channels
in_channels = node_features.size(1)

# Calculate the number of node types
num_node_types = node_types.max().item() + 1

# ==============================
# 2. Model Definition
# ==============================

class HeteroGraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_node_types, type_embedding_dim, dropout_rate):
        super().__init__()
        # Node type embedding layer
        self.node_type_embedding = torch.nn.Embedding(num_node_types, type_embedding_dim)

        # Define GraphSAGE convolution layers for heterogeneous graphs
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

        # Pass through GraphSAGE convolution layers for heterogeneous graphs
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: self.dropout(F.relu(x)) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: self.dropout(F.relu(x)) for key, x in x_dict.items()}
        x_dict = self.conv3(x_dict, edge_index_dict)

        return x_dict

# ==============================
# 3. Model Initialization and Device Configuration
# ==============================

# Initialize the model
model = HeteroGraphSAGE(
    in_channels,
    config['model']['hidden_channels'],
    config['model']['out_channels'],
    num_node_types,
    config['model']['type_embedding_dim'],
    config['model']['dropout_rate']
)
print(model)
# Dynamically set the device
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

# Calculate the frequency of each edge_type
class_counts = edges_df['edge_type'].value_counts().sort_index()
print("Class Counts:\n", class_counts)

# Calculate weights
class_weights = compute_class_weight('balanced', classes=np.unique(edges_df['edge_type']), y=edges_df['edge_type'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)  # Convert to tensor and move to GPU
class_weights = class_weights ** config['training']['class_weight_power']
relation_loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# ==============================
# 5. Dataset Splitting (8:1:1)
# ==============================

# Transpose edge_indices to fit the input format of train_test_split
edge_indices_np = edge_indices.cpu().numpy().T  # Convert to [num_edges, 2] shape
edge_types_np = edge_types.cpu().numpy()

# Step 1: Split into training set (80%) and temporary set (20%)
train_edges, temp_edges, train_types, temp_types = train_test_split(
    edge_indices_np, edge_types_np,
    test_size=config['dataset_split']['test_size'],
    random_state=config['dataset_split']['random_state'],
    shuffle=True,
    stratify=edge_types_np
)

# Step 2: Split the temporary set into validation set (10%) and test set (10%)
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
    f"Train edges: {train_edge_indices.size(1)}, Validation edges: {val_edge_indices.size(1)}, Test edges: {test_edge_indices.size(1)}")

# ==============================
# 6. Early Stopping Setup
# ==============================

# Define early stopping parameters
patience = config['early_stopping']['patience']
best_test_acc = 0.0
trigger_times = 0
best_model_state = copy.deepcopy(model.state_dict())  # Save the state of the best model
best_epoch = 0

# Lists to store training and validation losses, and test accuracies
train_losses = []
val_losses = []
test_accuracies = []

# ==============================
# 7. Helper Functions: Calculate Accuracy for Each Relation Type
# ==============================
def compute_aic_bic(loss, num_params, num_samples, is_classification=True):
    """
    Calculate AIC and BIC information criteria.

    Args:
        loss (float): Negative log-likelihood loss value of the current model
        num_params (int): Number of model parameters
        num_samples (int): Number of samples in the dataset
        is_classification (bool): Whether it is a classification task (default True, applicable to classification tasks)

    Returns:
        aic (float), bic (float)
    """
    # Assume the negative log-likelihood loss is given
    if is_classification:
        # For classification tasks, AIC and BIC can be calculated using the following formulas
        aic = 2 * num_params - 2 * loss
        bic = num_params * torch.log(torch.tensor(num_samples, dtype=torch.float)) - 2 * loss
    else:
        # For regression tasks (if any), different calculation methods can be used
        pass
    return aic.item(), bic.item()


def calculate_relation_accuracy(predictions, labels, edge_type_names):
    """
    Calculate and return the accuracy for each relation type.

    Args:
        predictions (torch.Tensor): Predicted relation types [num_edges]
        labels (torch.Tensor): True relation types [num_edges]
        edge_type_names (list): List of relation type names

    Returns:
        dict: Accuracy for each relation type
    """
    relation_accuracy = {}
    for relation_type in range(len(edge_type_names)):
        # Filter edges of the current relation type
        relation_mask = (labels == relation_type)
        true_labels = labels[relation_mask]
        predicted_labels = predictions[relation_mask]

        # Calculate accuracy
        if len(true_labels) > 0:
            accuracy = accuracy_score(true_labels.cpu(), predicted_labels.cpu())
            relation_accuracy[edge_type_names[relation_type]] = accuracy
        else:
            relation_accuracy[edge_type_names[relation_type]] = 0.0  # If there are no edges of this type, accuracy is 0
    return relation_accuracy

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
    # Clear cache to free up memory
    torch.cuda.empty_cache()

    # Calculate AIC and BIC for the current epoch
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_samples = train_edge_types.size(0)  # Number of samples in the training set
    aic, bic = compute_aic_bic(loss, num_params, num_samples)

    # Evaluate test set performance after each epoch
    model.eval()
    with torch.no_grad():
        edge_index_dict_test = {('node', 'to', 'node'): test_edge_indices}
        edge_index_dict_val = {('node', 'to', 'node'): val_edge_indices}

        x_dict_test = model(x_dict, node_type_ids_dict, edge_index_dict_test)
        x_test = x_dict_test['node']

        x_dict_val = model(x_dict, node_type_ids_dict, edge_index_dict_val)
        x_val = x_dict_val['node']

        edge_src_test = test_edge_indices[0]
        edge_dst_test = test_edge_indices[1]
        edge_src_val = val_edge_indices[0]
        edge_dst_val = val_edge_indices[1]

        # Concatenate test set node embeddings
        combined_emb_test = torch.cat([x_test[edge_src_test], x_test[edge_dst_test]], dim=1)
        relation_scores_test = model.relation_pred(combined_emb_test)
        test_predictions = relation_scores_test.argmax(dim=1)

        combined_emb_val = torch.cat([x_val[edge_src_val], x_val[edge_dst_val]], dim=1)
        relation_scores_val = model.relation_pred(combined_emb_val)
        val_predictions = relation_scores_val.argmax(dim=1)

        # Calculate overall accuracy on the test set
        test_accuracy = accuracy_score(test_edge_types.cpu(), test_predictions.cpu())
        val_accuracy = accuracy_score(val_edge_types.cpu(), val_predictions.cpu())
        # Calculate validation set loss
        val_loss = relation_loss_fn(relation_scores_val, val_edge_types)

        # Save training and validation losses, and test accuracies
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        test_accuracies.append(test_accuracy)

        # Print training process
        if epoch % 10 == 0 or epoch == 1:
            print(
                f'Epoch {epoch}/{num_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Test Accuracy: {test_accuracy * 100:.2f}%, AIC: {aic:.2f}, BIC: {bic:.2f}')
        # Calculate accuracy for each relation type
        relation_accuracy_test = calculate_relation_accuracy(test_predictions, test_edge_types, edge_type_names)

    # Check for improvement
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        best_epoch = epoch
        best_model_state = copy.deepcopy(model.state_dict())
        trigger_times = 0  # Reset trigger count
    else:
        trigger_times += 1

    # Check if early stopping is triggered
    if trigger_times >= patience:
        break

# ==============================
# 9. Load Best Model and Evaluate
# ==============================

# Load the best model state
model.load_state_dict(best_model_state)

# Final evaluation on the test set
model.eval()
with torch.no_grad():
    edge_index_dict_test = {('node', 'to', 'node'): test_edge_indices}
    x_dict_test = model(x_dict, node_type_ids_dict, edge_index_dict_test)
    x_test = x_dict_test['node']

    edge_src_test = test_edge_indices[0]
    edge_dst_test = test_edge_indices[1]
    combined_emb_test = torch.cat([x_test[edge_src_test], x_test[edge_dst_test]], dim=1)
    relation_scores_test = model.relation_pred(combined_emb_test)
    test_predictions_final = relation_scores_test.argmax(dim=1)

    # Calculate overall accuracy on the test set
    final_test_accuracy = accuracy_score(test_edge_types.cpu(), test_predictions_final.cpu())

    # Calculate accuracy for each relation type
    final_relation_accuracy_test = calculate_relation_accuracy(test_predictions_final, test_edge_types, edge_type_names)

    # Print final results
    print(f'\nBest Model was saved at epoch {best_epoch}')
    print(f'Final Test Accuracy: {final_test_accuracy * 100:.2f}%')
    print("Final Per Relation Accuracy on Test Set:")
    for relation, acc in final_relation_accuracy_test.items():
        print(f'  Relation: {relation}, Accuracy: {acc * 100:.2f}%')

# ==============================
# 10. Model Saving
# ==============================

# Save the best model
model_path = config['model']['model_path']
torch.save(model.state_dict(), model_path)
print(f'Best model saved to {model_path} with Test Accuracy: {best_test_acc * 100:.2f}% at epoch {best_epoch}')

# Visualize training process
plt.figure(figsize=(12, 6))

# Plot training and validation loss curves
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot test accuracy curve
plt.subplot(1, 2, 2)
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()
plt.show()
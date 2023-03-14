import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader

# Load the dataset
dataset_name = "citeseer"
dataset_loader = Dataset_Loader(dName=dataset_name)
dataset_loader.dataset_source_folder_path = "../../data/stage_5_data/" + dataset_name
data = dataset_loader.load()

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_features, hidden_dim, output_features, dropout):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(input_features)
        self.gc1 = nn.Linear(input_features, hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm3 = nn.BatchNorm1d(hidden_dim)
        self.gc3 = nn.Linear(hidden_dim, output_features)

    def forward(self, x, adj):
        x = self.gc1(torch.sparse.mm(adj, x))
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.gc2(torch.sparse.mm(adj, x))
        x = F.relu(x)
        x = self.dropout(x)
        x = self.norm3(x)
        x = self.gc3(torch.sparse.mm(adj, x))
        return F.log_softmax(x, dim=1)

# Train the model
def train(model, optimizer, criterion, features, adj, labels, train_idx):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = criterion(output[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluate the model on the validation set
def evaluate(model, features, adj, labels, val_idx):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        val_loss = criterion(output[val_idx], labels[val_idx])
        _, val_pred = output.max(dim=1)
        val_correct = int(val_pred[val_idx].eq(labels[val_idx]).sum().item())
        val_acc = val_correct / len(val_idx)
    return val_loss.item(), val_acc

# Evaluate the model on the test set
def test(model, features, adj, labels, test_idx):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        test_loss = criterion(output[test_idx], labels[test_idx])
        _, test_pred = output.max(dim=1)
        test_correct = int(test_pred[test_idx].eq(labels[test_idx]).sum().item())
        test_acc = test_correct / len(test_idx)
    return test_loss.item(), test_acc

import itertools

train_idx = data['train_test_val']['idx_train']
val_idx = data['train_test_val']['idx_val']
test_idx = data['train_test_val']['idx_test']

patience = 30
counter = 0
best_val_acc = 0
best_model = None
epochs = 500

# Define the hyperparameters to search over
dropout_vals = [0.2, 0.5, 0.1, 0.3, 0.7]
weight_decay_vals = [1e-4, 5e-4, 1e-5, 1e-6, 5e-6]
lr_vals = [1e-3, 1e-2, 1e-4, 5e-3, 1e-1]
optimizer_vals = [optim.Adam, optim.SGD, optim.Adagrad]
criterion_vals = [nn.CrossEntropyLoss(), nn.NLLLoss()]

# Create a list of all possible hyperparameter combinations
hyperparams = list(itertools.product(dropout_vals, weight_decay_vals, lr_vals, optimizer_vals, criterion_vals))

# Train and evaluate the model with each set of hyperparameters
results = {}
for i, (dropout, weight_decay, lr, optimizer, criterion) in enumerate(hyperparams):
    print('Training model {}/{}'.format(i+1, len(hyperparams)))
    model = GCN(input_features=data['graph']['X'].shape[1], hidden_dim=128, output_features=data['graph']['y'].max().item() + 1, dropout=dropout)
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        loss = train(model, optimizer, criterion, data['graph']['X'], data['graph']['utility']['A'], data['graph']['y'], train_idx)
        val_loss, val_acc = evaluate(model, data['graph']['X'], data['graph']['utility']['A'], data['graph']['y'], val_idx)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
        if counter == patience:
            break
    model.load_state_dict(best_model)
    test_loss, test_acc = test(model, data['graph']['X'], data['graph']['utility']['A'], data['graph']['y'], test_idx)
    results[(dropout, weight_decay, lr, str(optimizer), criterion.__class__.__name__)] = (test_loss, test_acc)
    print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(test_loss, test_acc))

# Print the results for each set of hyperparameters
for hyperparam, (test_loss, test_acc) in results.items():
    print('Hyperparams: {}, Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(hyperparam, test_loss, test_acc))
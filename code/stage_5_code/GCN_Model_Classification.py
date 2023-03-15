import torch
import torch.nn as nn
from sklearn import metrics


# Define the GCN model
# Define the GCN model
class GCN(nn.Module):
    def __init__(self, input_features, hidden_dim, output_features, dropout):
        super(GCN, self).__init__()
        self.leakyRelu = nn.ReLU()
        self.gc1 = GraphConvolution(input_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_features)
        self.bn2 = nn.BatchNorm1d(output_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.leakyRelu(self.bn1(self.gc1(x, adj)))
        x = self.dropout(x)
        x = self.bn2(self.gc2(x, adj))
        return nn.LogSoftmax(dim=1)(x)

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        output = output + self.bias
        return output

def evaluate_all_metrics(pred, true):
    f1 = metrics.f1_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy(), average="weighted", zero_division=0)
    accuracy = metrics.accuracy_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy())
    recall = metrics.recall_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy(), average="weighted", zero_division=0)
    precision = metrics.precision_score(pred.cpu().detach().numpy(), true.cpu().detach().numpy(), average="weighted", zero_division=0)
    return f1, accuracy, recall, precision

# Train the model
def train(model, optimizer, criterion, features, adj, labels, train_idx):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss = criterion(output[train_idx], labels[train_idx])
    loss.backward()
    # Exploding grad
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    _, predicted = output.max(dim=1)
    return loss.item(), evaluate_all_metrics(predicted[train_idx], labels[train_idx])


# Evaluate the model on the validation set
def evaluate(model, criterion,features, adj, labels, val_idx):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        val_loss = criterion(output[val_idx], labels[val_idx])
        _, predicted = output.max(dim=1)
    return val_loss.item(), evaluate_all_metrics(predicted[val_idx], labels[val_idx])

# Evaluate the model on the test set
def test(model, criterion, features, adj, labels, test_idx):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        test_loss = criterion(output[test_idx], labels[test_idx])
        _, predicted = output.max(dim=1)
    return test_loss.item(), evaluate_all_metrics(predicted[test_idx], labels[test_idx])
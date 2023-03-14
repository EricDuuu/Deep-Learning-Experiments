import torch.nn as nn
import torch.optim as optim

from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.GCN_Model_Classification import GCN, train, evaluate, test
from code.stage_5_code.Plotter import plot_acc_loss

# Load the dataset
dataset_name = "citeseer"
dataset_loader = Dataset_Loader(dName=dataset_name)
dataset_loader.dataset_source_folder_path = "../../data/stage_5_data/" + dataset_name
data = dataset_loader.load()

# Split the dataset into train, validation, and test sets
train_idx = data['train_test_val']['idx_train']
val_idx = data['train_test_val']['idx_val']
test_idx = data['train_test_val']['idx_test']

input_features = data['graph']['X']
adj_list = data['graph']['utility']['A']
labels = data['graph']['y']

# Initialize the model, optimizer, and loss function
epochs = 500
dropout = 0.5 # Best : 0.5
weight_decay = 5e-4 # Best: 5e-4
hidden_dim = 128
model = GCN(input_features=input_features.shape[1], hidden_dim=hidden_dim,
            output_features=labels.max().item()+1, dropout=dropout)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()


# Early stopping parameters
patience = 100
counter = 0
best_val_acc = 0
best_model = None
val_accs = []
val_losses = []

# Train the model with early stopping
for epoch in range(epochs):
    loss, train_eval = train(model, optimizer, criterion, input_features, adj_list, labels, train_idx)
    val_loss, val_eval = evaluate(model, criterion, input_features, adj_list, labels, val_idx)
    f1, accuracy, recall, precision = val_eval
    val_losses.append(val_loss)
    val_accs.append(accuracy)
    print(
        'Epoch {}/{}: Train Loss {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} '
        '| Val F1 {:.4f} | Val Recall {:.4f} | Val Precision {:.4f}'.format(
            epoch + 1, epochs, loss, val_loss, accuracy, f1, recall, precision))

    # Check if the model has improved on the validation set
    if accuracy > best_val_acc:
        best_val_acc = accuracy
        best_model = model.state_dict()
        counter = 0
    else:
        counter += 1

    # Stop training if the model has not improved for `patience` epochs
    if counter == patience:
        print('Early stopping: validation accuracy has not improved in {} epochs'.format(patience))
        break

# Load the best model and evaluate it on the test set
model.load_state_dict(best_model)
test_loss, test_eval = test(model, criterion, input_features, adj_list, labels, test_idx)
f1, accuracy, recall, precision = test_eval
print('Test Loss {:.4f} | Test Acc {:.4f} | Test F1 {:.4f} |'
      ' Test Recall {:.4f} | Test Precision {:.4f}'.format(
            test_loss, accuracy, f1, recall, precision))

plot_acc_loss("GCN", val_accs, val_loss)
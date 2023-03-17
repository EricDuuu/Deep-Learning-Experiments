import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from code.stage_5_code.Dataset_Loader_Node_Classification import Dataset_Loader
from code.stage_5_code.GCN_Model_Classification import GCN, train, evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(10)
torch.manual_seed(10)

# Load the dataset
dataset_name = "citeseer"
dataset_loader = Dataset_Loader(dName=dataset_name)
dataset_loader.dataset_source_folder_path = "../../data/stage_5_data/" + dataset_name
data = dataset_loader.load()

# Split the dataset into train, validation, and test sets
train_idx = data['train_test_val']['idx_train']
val_idx = data['train_test_val']['idx_val']
test_idx = data['train_test_val']['idx_test']

input_features = data['graph']['X'].to(device)
adj_list = data['graph']['utility']['A'].to(device)
labels = data['graph']['y'].to(device)

epochs = 500

from sklearn.model_selection import ParameterGrid

param_grid = {
    'dropout': [i / 10 for i in range(1, 10, 1)],
    'weight_decay': [0.001],
    'hidden_dim': [i for i in range(16, 129, 32)],
    'learning_rate': [i / 100.0 for i in range(1, 91, 2)]
}
grid = list(ParameterGrid(param_grid))

patience = 200
best_val_acc = 0
best_params = None
criterion = nn.CrossEntropyLoss().to(device)

for i, params in enumerate(grid):
    # set the hyperparameters for this iteration
    dropout = params['dropout']
    weight_decay = params['weight_decay']
    hidden_dim = params['hidden_dim']
    learning_rate = params['learning_rate']

    model = GCN(input_features=input_features.shape[1], hidden_dim=hidden_dim,
                output_features=labels.max().item() + 1, dropout=dropout)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)

    val_accs = []

    for epoch in range(epochs):
        loss, train_eval = train(model, optimizer, criterion,
                                 input_features,
                                 adj_list,
                                 labels,
                                 train_idx)

        val_loss, val_eval = evaluate(model,
                                      criterion,
                                      input_features,
                                      adj_list,
                                      labels,
                                      val_idx)

        f1, accuracy, recall, precision = val_eval
        val_accs.append(accuracy)

        if epoch > patience and np.mean(val_accs[-patience:]) <= np.mean(val_accs[-(patience + 1):-1]):
            break

    if np.mean(val_accs[-patience:]) > best_val_acc:
        best_val_acc = np.mean(val_accs[-patience:])
        best_params = params

    print('Progress: {}/{}'.format(i + 1, len(grid)) + str(params) + str(np.mean(val_accs[-patience:])))

print('Best validation accuracy:', best_val_acc)
print('Best hyperparameters:', best_params)
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torch.utils.data import TensorDataset, DataLoader

from code.stage_4_code.Dataset_Loader_Generation import Data_Loader
from code.stage_4_code.RNN_Model_Classification import LSTM, train


def generate_text(model, word2index, index2word, seed_sequence, max_length=20, randomness=1.0, k=30):
    model.eval()
    with torch.no_grad():
        input_sequence = seed_sequence.copy()
        for i in range(max_length):
            input_tensor = torch.LongTensor([word2index[w] for w in input_sequence[-sequence_length:]])
            input_tensor = input_tensor.unsqueeze(0).to(device)  # Turn into list of encoded ints
            output = model(input_tensor)
            logits = output.squeeze() / randomness  # randomness, else it will predict already known jokes
            # probs = nn.functional.gumbel_softmax(logits, tau=randomness, dim=-1)
            # probs = nn.functional.softmax(logits, dim=-1)
            probs = nn.functional.gumbel_softmax(logits, tau=randomness, dim=-1)

            top_k_probs, top_k_indices = torch.topk(probs, k)  # Topk frequent words for better randomization
            top_k_probs = top_k_probs / top_k_probs.sum()  # Normalize probabilities
            word_idx = torch.multinomial(top_k_probs, 1).item()

            word = index2word[top_k_indices[word_idx]]
            input_sequence.append(word)
            if word == '<EOJ>':
                break
    return ' '.join(input_sequence)


if 1:
    sequence_length = 3
    # Returns encoded and padded dataset
    train_data, vocab = Data_Loader(sequence_length).run()
    word2index = vocab.get_stoi()
    index2word = vocab.get_itos()

    input_sequences = []
    target_sequences = []
    for joke in train_data:
        for i in range(len(joke) - sequence_length):
            input_seq = joke[i:i + sequence_length]
            target_seq = joke[i + sequence_length]
            input_sequences.append(input_seq)
            target_sequences.append(target_seq)

    # Model Parameters
    batch_size = 32
    vocab_size = len(vocab)
    embedding_dim = 100  # GloVe: 50 100 200 300
    hidden_dim = 300
    output_dim = len(vocab)
    layers = 2
    bidirectional = True
    dropout_rate = 0.5

    # Initialize Dataloaders for ease of batching
    train_data = TensorDataset(torch.LongTensor(input_sequences), torch.LongTensor(target_sequences))
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Initialize Model
    model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, layers, bidirectional, dropout_rate)

    # Embed the vectors
    vectors = torchtext.vocab.GloVe(name="6B", dim=embedding_dim)
    pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
    model.embedding.weight.data = pretrained_embedding

    # Model Hyperparameters
    loadSavedModel = False
    model_name = "LSTM"
    epochs = 60
    lr = 0.005
    # loss functions: CrossEntropyLoss BCEWithLogitsLoss BCELoss
    loss_function = nn.CrossEntropyLoss()
    # optimizers: Adam SparseAdam Adamax ASGD NAdam RAdam SGD Adadelta Adagrad
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_function = loss_function.to(device)

    print("===TRAINING===")
    if loadSavedModel == False:
        losses, accuracies, precisions, F1_scores, recalls = train(epochs, train_dataloader, model,
                                                                   loss_function, optimizer, device)
    print("===TRAINING END===")

    if loadSavedModel == False:
        model_path = "joke_generator.pt"
        torch.save(model.state_dict(), model_path)
    else:
        model_path = "joke_generator.pt"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))

    # Generate text using the trained model
    seed_sequence = ['what', 'did', 'the']
    for i in range(20):
        generated_text = generate_text(model, word2index, index2word, seed_sequence, max_length=30, randomness=0.8)
        print(generated_text)
    seed_sequence = ['smell', 'you', 'later']
    for i in range(20):
        generated_text = generate_text(model, word2index, index2word, seed_sequence, max_length=30, randomness=2.4)
        print(generated_text)
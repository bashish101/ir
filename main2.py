# main file for CNN based retrieval model
import os
import numpy as np

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from conv_knrm import *
from data_reader import DataReader
from embedding_loader import create_embedding_matrix
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, data, n_epochs = 10, mini_batch_size = 16):
    # Use 80% data from training and 20% for validation 
    data_size = len(data)
    train_size = int(0.8 * len(data))
    val_size = data_size - train_size
    train_set, val_set = np.arange(0, train_size), np.arange(train_size, data_size)
    train_size = len(train_set) / mini_batch_size
    val_size = len(val_set) / mini_batch_size

    train_loader = DataLoader(data, 
                              batch_size = mini_batch_size,
                              sampler = SubsetRandomSampler(train_set),
                              num_workers = 1)
    val_loader = DataLoader(data, 
                            batch_size = mini_batch_size,
                            sampler = SubsetRandomSampler(val_set),
                            num_workers = 1)

    # Create loss function and optimizer
    criterion = HingeLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    
    chk_path = 'convknrm_chk.pth'
    best_loss = 9999.
    for epoch in range(n_epochs):
        # Training
        train_loss = 0.0
        for batch_idx, (query, pos, neg) in enumerate(train_loader):
            query = torch.stack(query, dim = 1).to(device)
            pos = torch.stack(pos, dim = 1).to(device)
            neg = torch.stack(neg, dim = 1).to(device)
            
            # Forward pass
            pos_score = model([query, pos]).view(-1)
            neg_score = model([query, neg]).view(-1)
            
            # Compute loss and backpropagate
            optimizer.zero_grad()
            loss = criterion(pos_score, neg_score)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                msg = 'Epoch: {0}  Training Batch: {1} Loss: {2:.2f}'.format(epoch, batch_idx, loss.item())
                print(msg, end = '\r')
    
        # Validation
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (query, pos, neg) in enumerate(val_loader):
                query = torch.stack(query, dim = 1).to(device)
                pos = torch.stack(pos, dim = 1).to(device)
                neg = torch.stack(neg, dim = 1).to(device)
            
                pos_score = model([query, pos]).view(-1)
                neg_score = model([query, neg]).view(-1)
                
                # Compute loss
                loss = criterion(pos_score, neg_score)
                
                val_loss += loss.item()
                            
                if batch_idx % 100 == 0:
                    msg = 'Epoch: {0} Validation Batch: {1} Loss: {2:.2f}'.format(epoch, batch_idx, loss.item())
                    print(msg, end = '\r')
                    
        # Display average loss values
        train_loss /= train_size
        val_loss /= val_size
        msg = '==> Epoch: {0}  Avg. Training Loss: {1:.2f} Avg. Validation Loss: {2:.2f}'.format(epoch, train_loss, val_loss)
        print(msg, end = '\n')
    
        # Save weights as checkpoints
        if val_loss < best_loss:
            best_loss = val_loss
            print('Saving checkpoint at ', chk_path)
            torch.save(model.state_dict(), chk_path)    
            
    print('Training complete!')
    
    return model
 
def test(model, query, pos_doc, neg_doc, weights_path = 'convknrm_chk.pth'):
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    
    x1 = torch.tensor([query], dtype = torch.long, device = device)
    x2 = torch.tensor([pos_doc], dtype = torch.long, device = device)
    x3 = torch.tensor([neg_doc], dtype = torch.long, device = device)
    
    y1 = model([x1, x2]).squeeze()
    y2 = model([x1, x3]).squeeze()
    return (y1, y2)
    
def main():    
    # Default settings
    vocab_size = 20000
    max_data_count = 10000
    pretrained_emb_path = './emb_weights.npy'
    data_path = './triples.train.small.tsv'
    
    # Load data  
    data_reader = DataReader(data_path = data_path,
                             data_count = max_data_count,
                             vocab_size = vocab_size)
    data = data_reader.load_data()  
    text_data = data_reader.unprocessed_data
    print('Data loaded successfully!')
    
    # Load pretrained embedding matrix (20000 X 300)
    if not os.path.exists(pretrained_emb_path):
        create_embedding_matrix(data_reader.word2idx_dict, data_reader.vocab, save_path = pretrained_emb_path)
    with open(pretrained_emb_path, 'rb') as fp:
        emb_weights = np.load(fp)
    print('Embeddings loaded successfully!')
    
    # Create a model
    model = ConvKNRM(vocab_size, emb_weights, n_filters = 64)
    model = model.to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    print('Trainable parameters: {}'.format(sum([np.prod(p.size()) for p in params])))
    print('Model created!')
    
    # Train the model
    train(model, data, n_epochs = 50) 

    # Test the model
    idx = 5000
    x1, x2, x3 = data[idx]
    scores = test(model, x1, x2, x3)
    print(' Query: {0} \n Positive Document: {1} \n Negative Document: {2} \n'.format(*text_data[idx]))
    print(' Fist document score = {0:.2f} \n Second document score = {1:.2f}'.format(*scores))
          
if __name__ == '__main__':
    main()

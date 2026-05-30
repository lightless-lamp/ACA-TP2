'''
This file trains the baseline/model.py. User can choose which dataset to train on (original or some augmented one).
'''

import torch
import torch.nn as nn
import torch.optim as optim
from src.baseline.model import BaselineCNN
import os
import copy

def train_model(train_loader, val_loader, device, epochs=10, lr=0.001, n_classes=75, stop=3):   
    model = BaselineCNN(num_classes=n_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    stop_counter = 0

    for epoch in range(epochs):

        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()*inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        
        # --- VALIDATION ---
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels)
                
                val_running_loss += val_loss.item()*val_inputs.size(0)
                
                _, val_predicted = torch.max(val_outputs.data, 1)
                total_val += val_labels.size(0)
                correct_val += (val_predicted == val_labels).sum().item()
        
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val

        # --- HISTORY ---
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{epochs} -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
    
        # --- EARLY STOP --- 
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict()) 
            stop_counter = 0
        else:
            stop_counter += 1
        
        if stop_counter >= stop:
            print(f"Early stopping at epoch {epoch+1}.")
            break

    model.load_state_dict(best_model_wts)

    return model, history
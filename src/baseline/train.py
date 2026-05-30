'''
This file trains the baseline/model.py. User can choose which dataset to train on (original or some augmented one).
'''

import torch
import torch.nn as nn
import torch.optim as optim
from src.baseline.model import BaselineCNN

def train_model(data_loader, device, epochs=10, lr=0.001, n_classes=75):
    train_loader = data_loader
    
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
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            # FALTA!!!
            history['val_loss'].append(0.0) 
            history['val_acc'].append(0.0)
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
        
    torch.save(model.state_dict(), 'src/baseline/baseline_weights.pth')
    return model, history
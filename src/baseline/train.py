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
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")
        
    torch.save(model.state_dict(), 'src/baseline/baseline_weights.pth')
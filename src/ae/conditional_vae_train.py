import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

def cvae_loss_function(recon_x, x, mu=None, logvar=None, function='mse', beta=1.0):        
    if function == 'mse':
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
    elif function == 'l1':
        recon_loss = F.l1_loss(recon_x, x, reduction='sum')
        
    else:
        raise ValueError(f"Função '{function}' não reconhecida. Escolha entre 'mse' ou 'l1'.")
    
    if mu is not None and logvar is not None:
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + (beta * kld_loss)
    else:
        kld_loss = torch.tensor(0.0, device=x.device)
        total_loss = recon_loss
    
    return total_loss, recon_loss, kld_loss


def train_cvae(model, train_loader, val_loader, device, epochs=20, lr=0.001, stop=5, beta=1.0, loss_function="mse"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    history = {
        'train_loss': [], 'train_recon': [], 'train_kld': [],
        'val_loss': [], 'val_recon': [], 'val_kld': []
    }
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    stop_counter = 0
    
    for epoch in range(epochs):
        # --- treino ---
        model.train()
        train_loss, train_recon, train_kld = 0.0, 0.0, 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # O CVAE recebe a imagem E a label no forward!
            reconstructed, mu, logvar = model(inputs, labels)

            # Calcula as perdas
            loss, recon, kld = cvae_loss_function(reconstructed, inputs, mu, logvar, loss_function, beta=beta)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon.item()
            train_kld += kld.item()
            
        # Normaliza as perdas pelo tamanho total do dataset (número de imagens)
        num_train_images = len(train_loader.dataset)
        epoch_train_loss = train_loss / num_train_images
        epoch_train_recon = train_recon / num_train_images
        epoch_train_kld = train_kld / num_train_images
        
        # --- validação ---
        model.eval()
        val_loss, val_recon, val_kld = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                
                val_reconstructed, val_mu, val_logvar = model(val_inputs, val_labels)
                
                v_loss, v_recon, v_kld = cvae_loss_function(val_reconstructed, val_inputs, val_mu, val_logvar, loss_function, beta=beta)
                
                val_loss += v_loss.item()
                val_recon += v_recon.item()
                val_kld += v_kld.item()
                
        num_val_images = len(val_loader.dataset)
        epoch_val_loss = val_loss / num_val_images
        epoch_val_recon = val_recon / num_val_images
        epoch_val_kld = val_kld / num_val_images
        
        # --- histórico ---
        history['train_loss'].append(epoch_train_loss)
        history['train_recon'].append(epoch_train_recon)
        history['train_kld'].append(epoch_train_kld)
        history['val_loss'].append(epoch_val_loss)
        history['val_recon'].append(epoch_val_recon)
        history['val_kld'].append(epoch_val_kld)
        
        print(f"Época {epoch+1}/{epochs} -> "
              f"Train Loss: {epoch_train_loss:.2f} (recon: {epoch_train_recon:.2f}, KLD: {epoch_train_kld:.2f}) || "
              f"Val Loss: {epoch_val_loss:.2f} (recon: {epoch_val_recon:.2f}, KLD: {epoch_val_kld:.2f})")
        
        # --- early stop ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            stop_counter = 0
        else:
            stop_counter += 1
            print(f"Val Loss did not improve")
            
        if stop_counter >= stop:
            print(f"Early stopping at epoch {epoch+1}.")
            break
            
    model.load_state_dict(best_model_wts)
    
    return model, history


def train_cae(model, train_loader, val_loader, device, epochs=20, lr=0.001, stop=5, 
              loss_function="mse", save_path="checkpoints", file_name="best_cae_weights.pth"):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
    
    history = {
        'train_loss': [], 'train_recon': [], 'train_kld': [],
        'val_loss': [], 'val_recon': [], 'val_kld': []
    }
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    stop_counter = 0
    
    for epoch in range(epochs):
        # --- Treino ---
        model.train()
        train_loss, train_recon = 0.0, 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            reconstructed = model(inputs, labels)

            loss, recon, _ = cvae_loss_function(reconstructed, inputs, mu=None, logvar=None, function=loss_function)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon.item()
            
        num_train_images = len(train_loader.dataset)
        epoch_train_loss = train_loss / num_train_images
        epoch_train_recon = train_recon / num_train_images
        
        # --- Validação ---
        model.eval()
        val_loss, val_recon = 0.0, 0.0
        
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                
                val_reconstructed = model(val_inputs, val_labels)
                
                v_loss, v_recon, _ = cvae_loss_function(val_reconstructed, val_inputs, mu=None, logvar=None, function=loss_function)
                
                val_loss += v_loss.item()
                val_recon += v_recon.item()
                
        num_val_images = len(val_loader.dataset)
        epoch_val_loss = val_loss / num_val_images
        epoch_val_recon = val_recon / num_val_images
        
        # --- History ---
        history['train_loss'].append(epoch_train_loss)
        history['train_recon'].append(epoch_train_recon)
        history['train_kld'].append(0.0)
        history['val_loss'].append(epoch_val_loss)
        history['val_recon'].append(epoch_val_recon)
        history['val_kld'].append(0.0)
        
        print(f"Época {epoch+1}/{epochs} -> Train Loss: {epoch_train_loss:.2f} || Val Loss: {epoch_val_loss:.2f}")
        
        # --- Early Stopping ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            stop_counter = 0
            
            if save_path:
                full_save_path = os.path.join(save_path, file_name)
                checkpoint = {
                    'model_state_dict': best_model_wts,
                    'history': history,
                    'epoch': epoch + 1
                }
                torch.save(checkpoint, full_save_path)
        else:
            stop_counter += 1
            
        if stop_counter >= stop:
            print(f"Early stopping at epoch {epoch+1}.")
            break
            
    model.load_state_dict(best_model_wts)
    return model, history
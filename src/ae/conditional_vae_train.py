import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def cvae_loss_function(recon_x, x, mu, logvar):
    """
    Calcula a perda combinada do VAE: Reconstrução + Divergência KL.
    """
    mse_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # 2. Divergência de Kullback-Leibler (KLD)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return mse_loss + kld_loss, mse_loss, kld_loss


def train_cvae(model, train_loader, val_loader, device, epochs=20, lr=0.001, stop=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    history = {
        'train_loss': [], 'train_bce': [], 'train_kld': [],
        'val_loss': [], 'val_bce': [], 'val_kld': []
    }
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    stop_counter = 0
    
    for epoch in range(epochs):
        # --- treino ---
        model.train()
        train_loss, train_bce, train_kld = 0.0, 0.0, 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # O CVAE recebe a imagem E a label no forward!
            reconstructed, mu, logvar = model(inputs, labels)
            
            # Calcula as perdas
            loss, bce, kld = cvae_loss_function(reconstructed, inputs, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_bce += bce.item()
            train_kld += kld.item()
            
        # Normaliza as perdas pelo tamanho total do dataset (número de imagens)
        num_train_images = len(train_loader.dataset)
        epoch_train_loss = train_loss / num_train_images
        epoch_train_bce = train_bce / num_train_images
        epoch_train_kld = train_kld / num_train_images
        
        # --- validação ---
        model.eval()
        val_loss, val_bce, val_kld = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs = val_inputs.to(device)
                val_labels = val_labels.to(device)
                
                val_reconstructed, val_mu, val_logvar = model(val_inputs, val_labels)
                
                v_loss, v_bce, v_kld = cvae_loss_function(val_reconstructed, val_inputs, val_mu, val_logvar)
                
                val_loss += v_loss.item()
                val_bce += v_bce.item()
                val_kld += v_kld.item()
                
        num_val_images = len(val_loader.dataset)
        epoch_val_loss = val_loss / num_val_images
        epoch_val_bce = val_bce / num_val_images
        epoch_val_kld = val_kld / num_val_images
        
        # --- histórico ---
        history['train_loss'].append(epoch_train_loss)
        history['train_bce'].append(epoch_train_bce)
        history['train_kld'].append(epoch_train_kld)
        history['val_loss'].append(epoch_val_loss)
        history['val_bce'].append(epoch_val_bce)
        history['val_kld'].append(epoch_val_kld)
        
        print(f"Época {epoch+1}/{epochs} -> "
              f"Train Loss: {epoch_train_loss:.2f} (BCE: {epoch_train_bce:.2f}, KLD: {epoch_train_kld:.2f}) || "
              f"Val Loss: {epoch_val_loss:.2f} (BCE: {epoch_val_bce:.2f}, KLD: {epoch_val_kld:.2f})")
        
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
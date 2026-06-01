import torch
import numpy as np
from src.utils.metrics2 import inception_score, frechet_inception_distance, mean_ssim

def evaluate(model, dataloader, device, num_imagens=256):
    model.eval()
    
    list_reais = []
    list_sinteticas = []
    
    print(f"Generating images")
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs, labels)
            
            if isinstance(outputs, tuple):
                reconstructed = outputs[0]
            else:
                reconstructed = outputs

            for i in range(inputs.size(0)):
                if len(list_reais) >= num_imagens:
                    break
                    
                img_real = inputs[i].cpu()
                img_fake = reconstructed[i].cpu()
                
                img_real_norm = (img_real - img_real.min()) / (img_real.max() - img_real.min() + 1e-8)
                img_fake_norm = (img_fake - img_fake.min()) / (img_fake.max() - img_fake.min() + 1e-8)
                
                list_reais.append(img_real_norm)
                list_sinteticas.append(img_fake_norm)
                
            if len(list_reais) >= num_imagens:
                break

    print("Calculating metrics")
    
    is_mean, is_std = inception_score(list_sinteticas, batch_size=32, splits=4, device=device)
    
    fid_value = frechet_inception_distance(list_reais, list_sinteticas, batch_size=32, device=device)
    
    ssim_value = mean_ssim(list_reais, list_sinteticas)
    
    print("\n================ Metrics ================")
    print(f"Inception Score (IS)                  : {is_mean:.4f} ± {is_std:.4f}  (Mais alto é melhor)")
    print(f"Frechet Inception Distance (FID)       : {fid_value:.4f}          (Mais baixo é melhor)")
    print(f"Structural Similarity Index (SSIM)     : {ssim_value:.4f}          (Mais próximo de 1 é melhor)")
    print("==========================================================")
    
    return {"is_mean": is_mean, "fid": fid_value, "ssim": ssim_value}
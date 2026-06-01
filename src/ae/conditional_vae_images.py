import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_cvae_losses(history, save=None):
    """
    Desenha os gráficos da evolução da Loss Total, MSE e KLD.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Treino')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validação')
    plt.title('Loss Total (Soma)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_bce'], 'g-', label='Reconstrução (MSE)')
    plt.plot(epochs, history['train_kld'], 'm-', label='Organização (KLD)')
    plt.title('Componentes da Loss (Treino)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        print(f"Loss plot saved in {save}")
        
    plt.show()
    
def comparar_duas_classes(model, device, classe_A, classe_B, class_names=None, num_samples=5, save=None):
    model.eval()
    
    imagens_A = model.generate_sample(device=device, label=classe_A, num_samples=num_samples)
    imagens_B = model.generate_sample(device=device, label=classe_B, num_samples=num_samples)
    
    imagens_A = imagens_A.cpu().permute(0, 2, 3, 1).numpy()
    imagens_B = imagens_B.cpu().permute(0, 2, 3, 1).numpy()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5))
    
    nome_A = class_names[classe_A] if class_names else f"Classe {classe_A}"
    nome_B = class_names[classe_B] if class_names else f"Classe {classe_B}"
    
    for i in range(num_samples):
        # O np.clip ou a normalização min-max manual ajuda a ajustar os valores do MSE para o plot
        img_A = (imagens_A[i] - imagens_A[i].min()) / (imagens_A[i].max() - imagens_A[i].min() + 1e-8)
        axes[0, i].imshow(img_A)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].text(-10, 32, nome_A, fontsize=12, fontweight='bold', 
                            ha='right', va='center', rotation=0)
            
    for i in range(num_samples):
        img_B = (imagens_B[i] - imagens_B[i].min()) / (imagens_B[i].max() - imagens_B[i].min() + 1e-8)
        axes[1, i].imshow(img_B)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].text(-10, 32, nome_B, fontsize=12, fontweight='bold', 
                            ha='right', va='center', rotation=0)
            
    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        print(f"Comparison saved in {save}")

    plt.show()

def comparar_reais_vs_sinteticas(model, device, dataloader, classe_A, classe_B, class_names=None, num_samples = 3, save=None):
    """
    Desenha uma grelha comparando Imagens Reais vs Sintéticas para duas classes.
    """
    model.eval()
    
    reais_A, reais_B = [], []
    
    for inputs, labels in dataloader:
        for img, lbl in zip(inputs, labels):
            if lbl.item() == classe_A and len(reais_A) < num_samples:
                reais_A.append(img)
            elif lbl.item() == classe_B and len(reais_B) < num_samples:
                reais_B.append(img)
        if len(reais_A) == num_samples and len(reais_B) == num_samples:
            break

    reais_A = torch.stack(reais_A).cpu().permute(0, 2, 3, 1).numpy()
    reais_B = torch.stack(reais_B).cpu().permute(0, 2, 3, 1).numpy()

    with torch.no_grad():
        sinteticas_A = model.generate_sample(device=device, label=classe_A, num_samples=num_samples)
        sinteticas_B = model.generate_sample(device=device, label=classe_B, num_samples=num_samples)
    
    sinteticas_A = sinteticas_A.cpu().permute(0, 2, 3, 1).numpy()
    sinteticas_B = sinteticas_B.cpu().permute(0, 2, 3, 1).numpy()

    fig, axes = plt.subplots(4, num_samples, figsize=(9, 10))
    
    nome_A = class_names[classe_A] if class_names else f"Classe {classe_A}"
    nome_B = class_names[classe_B] if class_names else f"Classe {classe_B}"
    
    def norm_img(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-8)

    for i in range(num_samples):
        axes[0, i].imshow(norm_img(reais_A[i]))
        axes[1, i].imshow(norm_img(sinteticas_A[i]))
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    axes[0, 0].text(-10, 32, f"{nome_A}\n(REAIS)", fontsize=11, fontweight='bold', ha='right', va='center', color='blue')
    axes[1, 0].text(-10, 32, f"{nome_A}\n(CVAE)", fontsize=11, fontweight='bold', ha='right', va='center', color='green')

    for i in range(num_samples):
        axes[2, i].imshow(norm_img(reais_B[i]))
        axes[3, i].imshow(norm_img(sinteticas_B[i]))
        axes[2, i].axis('off')
        axes[3, i].axis('off')
    axes[2, 0].text(-10, 32, f"{nome_B}\n(REAIS)", fontsize=11, fontweight='bold', ha='right', va='center', color='blue')
    axes[3, 0].text(-10, 32, f"{nome_B}\n(CVAE)", fontsize=11, fontweight='bold', ha='right', va='center', color='green')

    plt.tight_layout()

    if save:
        plt.savefig(save, bbox_inches='tight', dpi=300)
        print(f"Comparison saved in {save}")
    
    plt.show()
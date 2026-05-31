import torch.nn as nn
import torch

class ConditionalVAE(nn.Module):
    def __init__(self, img_channels=3, num_classes=75, latent_dim=128, img_size=64):
        super(ConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # 1. EMBEDDING PARA AS CLASSES
        # Transforma o ID da classe num vetor de tamanho 50
        self.class_embedding = nn.Embedding(num_classes, 50)
        
        # 2. ENCODER CONVOLUCIONAL
        # Entra: Imagem (3 canais) + Canal de classe (1 canal criado a partir do embedding) = 4 canais
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels + 1, 32, kernel_size=4, stride=2, padding=1), # -> img_size / 2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),              # -> img_size / 4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),             # -> img_size / 8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calcular o tamanho da saída do flatten baseado no tamanho da imagem
        flatten_size = 128 * (img_size // 8) * (img_size // 8)
        
        # Camadas lineares para a Média (Mu) e Variância (Log-Var)
        self.fc_mu = nn.Linear(flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(flatten_size, latent_dim)
        
        # 3. DECODER CONVOLUCIONAL
        # O Decoder recebe o Espaço Latente (latent_dim) + Condição da Classe (50)
        self.decoder_input = nn.Linear(latent_dim + 50, flatten_size)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Garante que os pixéis finais ficam entre [0, 1]
        )

    def reparameterize(self, mu, logvar):
        """O truque da reparametrização que permite o backpropagation num VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, labels):
        batch_size = x.size(0)
        
        # --- PROCESSO DO ENCODER ---
        # 1. Processar a label para o Encoder
        c = self.class_embedding(labels) # shape: [batch_size, 50]
        # Transforma o embedding num canal 2D preenchido [batch_size, 1, img_size, img_size]
        c_spatial = c.view(batch_size, 50, 1, 1).expand(batch_size, 50, self.img_size, self.img_size)
        # Vamos usar apenas 1 canal de forma simples projetando o embedding projetado
        c_channel = torch.mean(c_spatial, dim=1, keepdim=True) 
        
        # Concatenar a imagem com o canal da classe
        x_conditioned = torch.cat([x, c_channel], dim=1) # shape: [batch_size, 4, img_size, img_size]
        
        # Passar pelo encoder e extrair distribuição
        hidden = self.encoder(x_conditioned)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        # Amostragem do vetor latente z
        z = self.reparameterize(mu, logvar)
        
        # --- PROCESSO DO DECODER ---
        # Concatenar o vetor latente z com o embedding da classe c
        z_conditioned = torch.cat([z, c], dim=1) # shape: [batch_size, latent_dim + 50]
        
        # Expandir de volta para o formato de imagem convolucional
        dec_input = self.decoder_input(z_conditioned)
        dec_input = dec_input.view(batch_size, 128, self.img_size // 8, self.img_size // 8)
        
        # Reconstruir imagem
        reconstructed_x = self.decoder(dec_input)
        
        return reconstructed_x, mu, logvar

    def generate_sample(self, device, label, num_samples=1):
        self.eval()
        with torch.no_grad():
            # Gerar vetor aleatório Z vindo de uma distribuição normal pura
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Criar o tensor da classe que queres gerar
            labels = torch.tensor([label] * num_samples, dtype=torch.long).to(device)
            c = self.class_embedding(labels)
            
            # Combinar e decodificar
            z_conditioned = torch.cat([z, c], dim=1)
            dec_input = self.decoder_input(z_conditioned)
            dec_input = dec_input.view(num_samples, 128, self.img_size // 8, self.img_size // 8)
            
            generated_images = self.decoder(dec_input)
            return generated_images
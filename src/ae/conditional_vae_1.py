import torch
import torch.nn as nn

class ConditionalVAE(nn.Module):
    def __init__(self, img_channels=3, num_classes=75, latent_dim=128, img_size=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_classes = num_classes
        
        embedding_dim = 50 
        
        # --- ENCODER ---
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(img_channels + embedding_dim, 32, kernel_size=4, stride=2, padding=1), # Saída: 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),               # Saída: 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),              # Saída: 8x8
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # Como a imagem é reduzida 3 vezes por stride=2 (64 -> 32 -> 16 -> 8)
        # O tamanho do mapa de características final é 8x8
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        
        # --- DECODER ---
        # O Decoder recebe o vetor latente z (latent_dim) + o embedding da classe (embedding_dim)
        self.decoder_input = nn.Linear(latent_dim + embedding_dim, 128 * 8 * 8)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Saída: 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # Saída: 32x32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, img_channels, kernel_size=4, stride=2, padding=1), # Saída: 64x64
            nn.Tanh() # Devolve os pixéis normalizados (geralmente entre -1 e 1)
        )
        
    def encode(self, x, labels):
        c = self.label_embedding(labels)
        c_spatial = c.view(c.size(0), c.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        x_conditioned = torch.cat([x, c_spatial], dim=1)
        
        out = self.conv_block(x_conditioned)
        out = torch.flatten(out, start_dim=1)
        
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        c = self.label_embedding(labels)
        z_conditioned = torch.cat([z, c], dim=1)
        
        out = self.decoder_input(z_conditioned)
        out = out.view(-1, 128, 8, 8) # Reconstrói o formato para a Convolução Transposta
        return self.decoder(out)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

    def generate_sample(self, device, label, num_samples=1):
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            labels = torch.tensor([label] * num_samples, dtype=torch.long).to(device)
            
            generated_images = self.decode(z, labels)
            return generated_images
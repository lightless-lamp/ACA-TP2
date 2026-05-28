we-need-more-butterflies/
│
├── data/                   # Diretório para armazenar os datasets
│   ├── raw/                # Dataset original extraído (pasta Train com estrutura hierárquica e train.csv) [cite: 62, 63]
│   └── augmented/          # Dataset final contendo os dados originais + amostras geradas (deve ser entregue no final) [cite: 48, 92]
│
├── notebooks/              # Jupyter Notebooks para exploração
│   ├── 01_eda.ipynb        # Análise exploratória das imagens e distribuição das 75 classes [cite: 60]
│   └── 02_metricas.ipynb   # Testes iniciais com Inception Score, FID ou SSIM [cite: 86]
│
├── src/                    # Código-fonte principal do projeto
│   ├── baseline/           # Scripts do modelo CNN fornecido (não deve ter a arquitetura modificada) 
│   │   ├── train.py        # Script de treino do baseline
│   │   └── evaluate.py     # Script para gerar o CSV da submissão do Kaggle [cite: 98]
│   │
│   ├── generative/         # Modelos para Data Augmentation [cite: 50]
│   │   ├── autoencoders/   # Implementação de AutoEncoders [cite: 52]
│   │   ├── gans/           # Implementação de Generative Adversarial Networks [cite: 53]
│   │   └── diffusion/      # (Opcional) Implementação de modelos de Difusão [cite: 54]
│   │
│   └── utils/              # Funções auxiliares (carregamento de dados, transformações, cálculo de métricas)
│
├── reports/                # Documentação e relatório final
│   ├── figures/            # Gráficos gerados para o relatório (ex: exemplos das imagens geradas)
│   └── paper/              # Arquivos LATEX ou Word no formato Springer LNCS (limite de 10 páginas) [cite: 16, 17]
│
├── requirements.txt        # Dependências do projeto para garantir replicação total 
├── .gitignore              # Arquivo para ignorar pastas de dados muito pesadas no versionamento (git)
└── README.md               # Descrição do projeto, como configurar o ambiente e como executar o pipeline [cite: 46]
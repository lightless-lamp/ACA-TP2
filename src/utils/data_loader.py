import os
import torch
from PIL import Image
from torch.utils import data

class ButterflyDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None, target_classes=None):
        self.img_labels = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        # ---------------------------
        if target_classes is not None:
            df = df[df['label'].isin(target_classes)]
            
        self.img_labels = df.reset_index(drop=True)
        # ---------------------------
    
        self.classes = sorted(self.img_labels['label'].unique())
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        label_name = self.img_labels.iloc[idx]['label']
        label_idx = self.class_to_idx[label_name]
        label = torch.tensor(label_idx, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label
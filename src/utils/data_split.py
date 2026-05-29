import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.data_loader import ButterflyDataset

def create_save_data_split(
        df, output_dir='data/splits', r_seed=42,
        test_size = 0.2, val_size = 0.24):
    os.makedirs(output_dir, exist_ok=True)
    required_files = ['train_split.csv', 'val_split.csv', 'test_split.csv', 'full_train_split.csv']

    if all(os.path.exists(os.path.join(output_dir, f)) for f in required_files):
        print(f"Splits already exist; did not create new ones.")
        return

    full_train_df, test_df = train_test_split(
        df,
        test_size = test_size,
        stratify=df['label'],
        random_state=r_seed
    )

    train_df, val_df = train_test_split(
        full_train_df,
        test_size = val_size,
        stratify=full_train_df['label'],
        random_state=r_seed
    )

    train_df.to_csv(os.path.join(output_dir, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'val_split.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test_split.csv'), index=False)
    full_train_df.to_csv(os.path.join(output_dir, 'full_train_split.csv'), index=False)
    
    print("Created and saved data splits.")
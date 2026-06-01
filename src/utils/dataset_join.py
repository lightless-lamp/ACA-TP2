import os
import pandas as pd
from src.config import TARGET_CLASSES

def join_datasets(base_augmented_dir, output_csv_name='train_augmented.csv'):
    csv_original_path = '../data/raw/train.csv'         
    df_original = pd.read_csv(csv_original_path)

    novas_linhas = []

    for class_idx, class_name in enumerate(TARGET_CLASSES):
        class_folder = os.path.join(base_augmented_dir, class_name)
        if os.path.exists(class_folder):
            for img_name in os.listdir(class_folder):
                if img_name.endswith('.png'):
                    img_path = os.path.join(class_folder, img_name)
                    
                    novas_linhas.append({
                        'filename': img_path, 
                        'label': class_name,
                    })

    df_augmented = pd.DataFrame(novas_linhas)
    df_final = pd.concat([df_original, df_augmented], ignore_index=True)

    os.makedirs(base_augmented_dir, exist_ok=True)
    output_path = os.path.join(base_augmented_dir, output_csv_name)
    df_final.to_csv(output_path, index=False)
        
    return output_path
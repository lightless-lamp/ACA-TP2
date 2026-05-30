import os
import json
import torch
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def save_experiment(model, history, config, base_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Pesos
    weights_path = os.path.join(run_dir, 'model_weights.pth')
    torch.save(model.state_dict(), weights_path)
    
    # Config
    config_to_save = config.copy()
    config_to_save['timestamp'] = timestamp
    
    config_path = os.path.join(run_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, indent=4, ensure_ascii=False)
    
    # History
    if history:
        history_df = pd.DataFrame(history)
        history_path = os.path.join(run_dir, 'training_history.csv')
        history_df.to_csv(history_path, index_label='epoch')
                
    return run_dir
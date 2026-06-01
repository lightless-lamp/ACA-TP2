import os
import json
import torch
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def save_cvae_experiment(model, history, config, base_dir, metrics=None, excel_name="overview_experiencias_cvae.xlsx"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_cvae_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    weights_path = os.path.join(run_dir, 'cvae_model_weights.pth')
    torch.save(model.state_dict(), weights_path)
    
    config_to_save = config.copy()
    config_to_save['timestamp'] = timestamp
    if metrics:
        config_to_save['final_metrics'] = metrics
        
    config_path = os.path.join(run_dir, 'config_and_metrics.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_to_save, f, indent=4, ensure_ascii=False)

    final_train_loss = None
    final_val_loss = None
    epochs_executed = 0
    
    if history:
        history_df = pd.DataFrame(history)
        history_path = os.path.join(run_dir, 'training_history.csv')
        history_df.to_csv(history_path, index_label='epoch')
        
        epochs_executed = len(history_df)
        if 'train_loss' in history_df.columns:
            final_train_loss = history_df['train_loss'].iloc[-1]
        if 'val_loss' in history_df.columns:
            final_val_loss = history_df['val_loss'].iloc[-1]
            
        plt.figure(figsize=(10, 5))
        if 'train_loss' in history_df.columns:
            plt.plot(history_df['train_loss'], label='Train Loss')
        if 'val_loss' in history_df.columns:
            plt.plot(history_df['val_loss'], label='Val Loss')
        plt.title('CVAE Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(run_dir, 'loss_curve.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()

    new_run_data = {
        "Run ID": f"run_cvae_{timestamp}",
        "Timestamp": timestamp,
        "Architecture": config.get("model_architecture", "ConditionalVAE"),
        "Latent Dim": config.get("latent_dim", None),
        "Loss Function": config.get("loss_function", None),
        "Epochs Executed": epochs_executed,
        "Final Train Loss": final_train_loss,
        "Final Val Loss": final_val_loss,
        "Inception Score (IS)": metrics.get("Inception Score (IS)", None) if metrics else None,
        "FID": metrics.get("Frechet Inception Distance (FID)", None) if metrics else None,
        "SSIM": metrics.get("Structural Similarity Index (SSIM)", None) if metrics else None
    }
    
    excel_path = os.path.join(base_dir, excel_name)
    
    if os.path.exists(excel_path):
        try:
            df_global = pd.read_excel(excel_path, sheet_name='Overview Geral')
            df_global = pd.concat([df_global, pd.DataFrame([new_run_data])], ignore_index=True)
        except Exception:
            df_global = pd.DataFrame([new_run_data])
    else:
        df_global = pd.DataFrame([new_run_data])

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_global.to_excel(writer, sheet_name='Overview Geral', index=False)
        
        worksheet = writer.sheets['Overview Geral']
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or '')) for cell in col)
            col_letter = col[0].column_letter
            worksheet.column_dimensions[col_letter].width = max(max_len + 3, 12)
            
    
    return run_dir
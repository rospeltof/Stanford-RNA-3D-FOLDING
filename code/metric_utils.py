from tensorflow.keras.callbacks import Callback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_tm_score(y_true, y_pred):
    L_ref = y_true.shape[0]
    d0 = 0.6 * np.sqrt(L_ref - 0.5) - 2.5 if L_ref >= 30 else 0.3
    distances = np.linalg.norm(y_true - y_pred, axis=1)
    return np.mean(1 / (1 + (distances / d0) ** 2))

def evaluate_tm_scores_by_sequence(df_true, df_pred):
    """
    df_true, df_pred: DataFrames con columnas [sequence_id, position, x_1, y_1, z_1]
    """
    tm_scores = []
    for seq_id in df_true['sequence_id'].unique():
        true_coords = df_true[df_true['sequence_id'] == seq_id][['x_1', 'y_1', 'z_1']].to_numpy()
        pred_coords = df_pred[df_pred['sequence_id'] == seq_id][['x_1', 'y_1', 'z_1']].to_numpy()
        
        if len(true_coords) != len(pred_coords):
            continue  # evitar errores si hay mismatch
        
        tm = compute_tm_score(true_coords, pred_coords)
        tm_scores.append(tm)
    
    return np.mean(tm_scores)

def plot_training_history(history, train_tm_score=None, val_tm_score=None):
    # === 1. Gráfica de Loss ===
    history_df = pd.DataFrame(history.history)
    epochs = range(1, len(history_df) + 1)

    plt.figure(figsize=(15, 5), dpi=300)
    sns.lineplot(x=epochs, y=history_df['loss'], marker='o', color='#118AB2', label='Pérdida entrenamiento')
    sns.lineplot(x=epochs, y=history_df['val_loss'], marker='o', color='#52414C', label='Pérdida validación')
    plt.title('Evolución de la pérdida por época', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Época', fontsize=13)
    plt.ylabel('Pérdida (Loss)', fontsize=13)
    plt.grid(True)
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.show()

    # === 2. Gráfica de TM-score ===
    if train_tm_score is not None and val_tm_score is not None:
        df_tm = pd.DataFrame({
            'Época': epochs,
            'TM-score entrenamiento': train_tm_score,
            'TM-score validación': val_tm_score
        })

        plt.figure(figsize=(15, 5), dpi=300)
        sns.lineplot(data=df_tm, x='Época', y='TM-score entrenamiento', marker='o', color='#118AB2', label='Entrenamiento')
        sns.lineplot(data=df_tm, x='Época', y='TM-score validación', marker='o', color='#DFBBB1', label='Validación')
        plt.title('Evolución del TM-score por época', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Época', fontsize=13)
        plt.ylabel('TM-score', fontsize=13)
        plt.grid(True)
        plt.legend()
        sns.despine()
        plt.tight_layout()
        plt.show()


class TMScoreCallback(Callback):
    def __init__(self, X_train, y_train, train_meta, X_val, y_val, val_meta):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.train_meta = train_meta  # debe tener ['sequence_id', 'position']
        self.X_val = X_val
        self.y_val = y_val
        self.val_meta = val_meta

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = self.model.predict(self.X_train, verbose=0)
        y_val_pred = self.model.predict(self.X_val, verbose=0)

        df_train_pred = self.train_meta.copy()
        df_val_pred = self.val_meta.copy()

        df_train_pred[['x_1', 'y_1', 'z_1']] = y_train_pred
        df_val_pred[['x_1', 'y_1', 'z_1']] = y_val_pred

        df_train_true = self.train_meta.copy()
        df_train_true[['x_1', 'y_1', 'z_1']] = self.y_train

        df_val_true = self.val_meta.copy()
        df_val_true[['x_1', 'y_1', 'z_1']] = self.y_val

        tm_train = evaluate_tm_scores_by_sequence(df_train_true, df_train_pred)
        tm_val = evaluate_tm_scores_by_sequence(df_val_true, df_val_pred)

        print(f"\nEpoch {epoch+1} — TM-score Train: {tm_train:.4f}, Val: {tm_val:.4f}")
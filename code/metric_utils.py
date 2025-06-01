import tensorflow as tf
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
    sns.lineplot(x=epochs, y=history_df['val_loss'], marker='o', color='#FF220C', label='Pérdida validación')
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
        sns.lineplot(data=df_tm, x='Época', y='TM-score validación', marker='o', color='#FF220C', label='Validación')
        plt.title('Evolución del TM-score por época', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Época', fontsize=13)
        plt.ylabel('TM-score', fontsize=13)
        plt.grid(True)
        plt.legend()
        sns.despine()
        plt.tight_layout()
        plt.show()


class TMScoreLogger(tf.keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        y_train_pred = self.model.predict(self.X_train, verbose=0)
        y_val_pred = self.model.predict(self.X_val, verbose=0)

        train_score = compute_tm_score(self.y_train, y_train_pred)
        val_score = compute_tm_score(self.y_val, y_val_pred)

        tm_train_scores.append(train_score)
        tm_val_scores.append(val_score)

        print(f"Epoch {epoch + 1} — TM-score Train: {train_score:.4f}, Val: {val_score:.4f}")
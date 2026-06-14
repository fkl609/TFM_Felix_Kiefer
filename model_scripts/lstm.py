"""
Modelo LSTM (Encoder-Decoder) a nivel de carácter.
Arquitectura adaptada para predicción secuencial (12 notas de entrada -> 4 notas de salida).
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

# Ocultar mensajes de información de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# ==========================================
# 0. CONFIGURACIÓN DE REPRODUCIBILIDAD Y RUTAS
# ==========================================
SEED = 31
np.random.seed(SEED)
tf.random.set_seed(SEED)

MODELS_DIR = '../models'
PLOTS_DIR = '../results/plots'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# 1. CARGA Y PREPARACIÓN DE DATOS
# ==========================================
with open('../dataset/input.txt', 'r', encoding='utf-8') as f:
    songs = [line.strip() for line in f.readlines() if line.strip()]

all_text = ''.join(songs)
chars = sorted(list(set(all_text)))
vocab_size = len(chars)

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

seq_length = 12   # Longitud del contexto de entrada
pred_length = 4   # Longitud de la secuencia de salida

X_data = []
y_data = []

for song in songs:
    for i in range(0, len(song) - seq_length - pred_length + 1):
        seq_in = song[i : i + seq_length]
        seq_out = song[i + seq_length : i + seq_length + pred_length]
        
        X_data.append([char_to_ix[char] for char in seq_in])
        y_data.append([char_to_ix[char] for char in seq_out])

# Transformación a One-Hot Encoding
X = to_categorical(X_data, num_classes=vocab_size)
y = to_categorical(y_data, num_classes=vocab_size) 

# ==========================================
# 2. DEFINICIÓN DEL MODELO
# ==========================================
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(seq_length, vocab_size)))
    
    hp_units = hp.Int('hidden_size', min_value=25, max_value=125, step=25)
    
    model.add(LSTM(hp_units))
    model.add(RepeatVector(pred_length))
    model.add(LSTM(hp_units, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1])
    opt = Adam(learning_rate=hp_learning_rate)
    
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# ==========================================
# 3. CLASES AUXILIARES Y CALLBACKS
# ==========================================
class LSTM_Tuner(kt.BayesianOptimization):
    """Subclase de Keras Tuner para inyectar dinámicamente el batch_size usando Optimización Bayesiana."""
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=[15, 30, 60])
        return super().run_trial(trial, *args, **kwargs)

class TimeHistory(Callback):
    """Callback para monitorizar los tiempos de ejecución del entrenamiento por época."""
    def on_train_begin(self, logs={}):
        self.total_start_time = time.time()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_times.append(time.time() - self.epoch_start_time)
        
    def on_train_end(self, logs={}):
        self.total_time = time.time() - self.total_start_time
        self.avg_time = sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0

# ==========================================
# 4. CONFIGURACIÓN DEL SINTONIZADOR Y ENTRENAMIENTO
# ==========================================
if __name__ == "__main__":
    safe_dir = os.path.join(os.path.expanduser('~'), 'tuner_results')

    tuner = LSTM_Tuner(
        build_model,
        objective='loss',
        max_trials=15,
        seed=SEED,
        directory=safe_dir,
        project_name='lstm_songs_bayes',
        overwrite=True
    )

    # Early stopping para evitar entrenar durante demasiadas épocas
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=30,
        min_delta=0,
        restore_best_weights=True
    )

    time_callback = TimeHistory()

    print("--- INICIANDO BÚSQUEDA DE HIPERPARÁMETROS ---")
    tuner.search(X, y, epochs=2000, callbacks=[early_stopping], verbose=1)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_size = best_hps.get('hidden_size')
    best_lr = best_hps.get('learning_rate')
    best_batch = best_hps.get('batch_size')

    print("--- OPTIMIZACIÓN FINALIZADA ---")
    print(f"Entrenando modelo final con -> H_Size: {best_size}, LR: {best_lr}, Batch: {best_batch}")

    best_model = tuner.hypermodel.build(best_hps)
    epochs_final = 10000

    # Añadimos el callback personalizado de tiempo a la lista
    history = best_model.fit(X, y, epochs=epochs_final, batch_size=best_batch, callbacks=[early_stopping, time_callback], verbose=1)

    print(f"\nEntrenamiento Finalizado. Tiempo Total: {time_callback.total_time:.2f}s | Tiempo Medio/Epoch: {time_callback.avg_time:.4f}s")

    # ==========================================
    # 5. GUARDADO DE RESULTADOS, MODELO Y METADATOS
    # ==========================================
    lr_str = str(best_lr).replace('.', '')
    base_filename = f"lstm_opt_{best_size}_{lr_str}_{best_batch}"

    # Generación y guardado de la gráfica de pérdida
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss', color='tab:blue')
    plt.xlabel("Epochs", fontweight='bold')
    plt.ylabel("Loss", fontweight='bold')
    plt.title(f"Curva de Aprendizaje (LSTM)\nH_Size: {best_size} | LR: {best_lr} | Batch: {best_batch}", fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{base_filename}.png"), dpi=300)
    plt.close()

    best_model.save(os.path.join(MODELS_DIR, f"{base_filename}.keras"))

    model_data = {
        'char_to_ix': char_to_ix,
        'ix_to_char': ix_to_char,
        'hidden_size': best_size,
        'vocab_size': vocab_size,
        'seq_length': seq_length,
        'pred_length': pred_length,
        'learning_rate': best_lr,
        'batch_size': best_batch,
        'epochs': len(history.history['loss']),
        'epoch_times': time_callback.epoch_times,
        'total_train_time': time_callback.total_time,
        'avg_epoch_time': time_callback.avg_time
    }

    with open(os.path.join(MODELS_DIR, f"{base_filename}.pkl"), 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\nModelo, gráfica y metadatos guardados correctamente bajo el prefijo: '{base_filename}'")
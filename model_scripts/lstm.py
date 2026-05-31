"""
Modelo LSTM (Encoder-Decoder) a nivel de carácter.
Arquitectura adaptada para predicción secuencial (12 notas de entrada -> 4 notas de salida).
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Ocultar mensajes de información de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# ==========================================
# 0. CONFIGURACIÓN DE REPRODUCIBILIDAD
# ==========================================
SEED = 31
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
# 3. CONFIGURACIÓN DEL SINTONIZADOR PERSONALIZADO
# ==========================================
class LSTM_Tuner(kt.RandomSearch):
    """Subclase de Keras Tuner para inyectar dinámicamente el batch_size."""
    def run_trial(self, trial, *args, **kwargs):
        kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', values=[15, 30, 60])
        return super().run_trial(trial, *args, **kwargs)

safe_dir = os.path.join(os.path.expanduser('~'), 'tuner_results')

tuner = LSTM_Tuner(
    build_model,
    objective='loss',
    max_trials=6,
    seed=SEED,
    directory=safe_dir,
    project_name='lstm_songs',
    overwrite=True
)

early_stopping = EarlyStopping(
    monitor='loss',
    patience=30,
    min_delta=0,
    restore_best_weights=True
)

# ==========================================
# 4. ENTRENAMIENTO Y BÚSQUEDA
# ==========================================
print("--- INICIANDO BÚSQUEDA DE HIPERPARÁMETROS ---")
tuner.search(X, y, epochs=2000, callbacks=[early_stopping], verbose=1)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_size = best_hps.get('hidden_size')
best_lr = best_hps.get('learning_rate')
best_batch = best_hps.get('batch_size')

print("\n-------------------------------------------------")
print(f"Mejor tamaño LSTM: {best_size}")
print(f"Mejor Learning Rate: {best_lr}")
print(f"Mejor Batch Size: {best_batch}")
print("-------------------------------------------------\n")

print("Entrenando modelo final optimizado...")
best_model = tuner.hypermodel.build(best_hps)
epochs_final = 10000

history = best_model.fit(X, y, epochs=epochs_final, batch_size=best_batch, callbacks=[early_stopping], verbose=1)

# ==========================================
# 5. GUARDADO DE RESULTADOS, MODELO Y METADATOS
# ==========================================
lr_str = str(best_lr).replace('.', '')
base_filename = f"lstm_model_opt_{best_size}_{lr_str}_{best_batch}"

# Generación y guardado de la gráfica de pérdida
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f'../results/plots/{base_filename}.png')
plt.show()

best_model.save(f'../models/{base_filename}.keras')

model_data = {
    'char_to_ix': char_to_ix,
    'ix_to_char': ix_to_char,
    'hidden_size': best_size,
    'vocab_size': vocab_size,
    'seq_length': seq_length,
    'pred_length': pred_length,
    'learning_rate': best_lr,
    'batch_size': best_batch,
    'epochs': len(history.history['loss']) # Se guarda la época real donde paró por el EarlyStopping
}

with open(f'../models/{base_filename}.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModelo, gráfica y metadatos guardados correctamente bajo el prefijo: '{base_filename}'")
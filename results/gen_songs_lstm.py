"""
Script de inferencia y evaluación para el modelo LSTM (Encoder-Decoder).
Evalúa la precisión de la predicción en modo Autorregresivo y Guiado.
"""

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 0. CONFIGURACIÓN DE RUTAS BASE
# ==========================================
DATASET_FILE = '../dataset/input.txt'
NAMES_FILE = '../dataset/song_names.txt'
OUTPUT_DIR = 'outputs'
PLOTS_DIR = 'plots'
RESULTS_CSV = 'results_lstm.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# 1. CARGA DE DATOS Y DEL ÚLTIMO MODELO EN PICKLE
# ==========================================
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    songs = [line.strip() for line in f if line.strip()]

with open(NAMES_FILE, 'r', encoding='utf-8') as f:
    song_names = [line.strip() for line in f if line.strip()]

def get_latest_model(models_dir='../models/', extension='lstm_*.pkl'):
    search_pattern = os.path.join(models_dir, extension)
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        raise FileNotFoundError(f"No se encontraron metadatos en '{models_dir}'.")
    return max(list_of_files, key=os.path.getmtime)

model_file = get_latest_model()
print(f"Cargando modelo: {model_file}")

with open(model_file, 'rb') as f:
    meta = pickle.load(f)

char_to_ix = meta['char_to_ix']
ix_to_char = meta['ix_to_char']
hidden_size = meta['hidden_size']
vocab_size = meta['vocab_size']
seq_length = meta['seq_length']
pred_length = meta['pred_length']
learning_rate = meta['learning_rate']
epochs = meta['epochs']
batch_size = meta.get('batch_size', 'N/A') 

total_train_time = meta.get('total_train_time', 'N/A')
avg_epoch_time = meta.get('avg_epoch_time', 'N/A')

lr_str = str(learning_rate).replace('.', '')
output_filename = f"lstm_{hidden_size}_{lr_str}_{batch_size}.txt"
output_path = os.path.join(OUTPUT_DIR, output_filename)
plot_filename = f"lstm_eval_{hidden_size}_{lr_str}_{batch_size}.png"
plot_path = os.path.join(PLOTS_DIR, plot_filename)

# ==========================================
# 2. CARGA DEL MODELO KERAS
# ==========================================
model_keras_file = model_file.replace('.pkl', '.keras')
model = load_model(model_keras_file)

# ==========================================
# 3. FUNCIONES DE GENERACIÓN
# ==========================================
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def generate_song_ar(model, seed_phrase, generate_size):
    """Generación Autorregresiva"""
    resultado = seed_phrase
    iterations = int(np.ceil(generate_size / pred_length))
    for _ in range(iterations):
        seq_in = resultado[-seq_length:]
        x_pred = to_categorical([[char_to_ix[c] for c in seq_in]], num_classes=vocab_size)
        pred_probs = model(x_pred, training=False)[0].numpy() 
        new_chars = [ix_to_char[np.argmax(pred_probs[i])] for i in range(pred_length)]
        resultado += "".join(new_chars)
    return resultado

def generate_song_guided(model, true_song):
    """Generación Guiada"""
    resultado = true_song[:seq_length]
    for i in range(0, len(true_song) - seq_length, pred_length):
        seq_in = true_song[i : i + seq_length]
        if len(seq_in) < seq_length: break
        x_pred = to_categorical([[char_to_ix[c] for c in seq_in]], num_classes=vocab_size)
        pred_probs = model(x_pred, training=False)[0].numpy()
        new_chars = [ix_to_char[np.argmax(pred_probs[j])] for j in range(pred_length)]
        resultado += "".join(new_chars)
    return resultado[:len(true_song)]

# ==========================================
# 4. EVALUACIÓN Y EXPORTACIÓN
# ==========================================
input_size = seq_length 
output_log = ''
total_chars = 0
acc_ar_total, acc_gui_total = 0, 0
lev_ar_total, lev_gui_total = 0, 0
total_time_ar, total_time_gui = 0, 0

song_acc_ar, song_acc_gui = [], []
song_lev_ar, song_lev_gui = [], []

print("\nGenerando predicciones...\n")

for i in range(len(songs)):
    song = songs[i]
    song_name = song_names[i]
    input_phrase = song[:input_size]
    size_to_generate = len(song) - input_size
    
    # Inferencia Autorregresiva
    start_ar = time.time()
    pred_ar = generate_song_ar(model, input_phrase, size_to_generate)[:len(song)]
    time_ar = time.time() - start_ar
    
    # Inferencia Guiada
    start_gui = time.time()
    pred_gui = generate_song_guided(model, song)[:len(song)]
    time_gui = time.time() - start_gui
    
    true_suffix = song[input_size:]
    pred_suffix_ar = pred_ar[input_size:]
    pred_suffix_gui = pred_gui[input_size:]
    
    match_ar = sum([true_suffix[j] == pred_suffix_ar[j] for j in range(len(true_suffix))])
    match_gui = sum([true_suffix[j] == pred_suffix_gui[j] for j in range(len(true_suffix))])
    
    acc_ar = match_ar / size_to_generate
    acc_gui = match_gui / size_to_generate
    
    lev_ar = levenshtein_distance(true_suffix, pred_suffix_ar) / len(true_suffix) if true_suffix else 0.0
    lev_gui = levenshtein_distance(true_suffix, pred_suffix_gui) / len(true_suffix) if true_suffix else 0.0
    
    song_acc_ar.append(acc_ar)
    song_acc_gui.append(acc_gui)
    song_lev_ar.append(lev_ar)
    song_lev_gui.append(lev_gui)
    
    acc_ar_total += match_ar
    acc_gui_total += match_gui
    total_chars += size_to_generate
    lev_ar_total += lev_ar
    lev_gui_total += lev_gui
    
    total_time_ar += time_ar
    total_time_gui += time_gui
    
    output_log += f"{song_name}\n"
    output_log += f"Expected:   {song}\n"
    output_log += f"Pred (AR):  {pred_ar}\n"
    output_log += f"Pred (GUI): {pred_gui}\n"
    output_log += f"Acc (AR/GUI): {acc_ar:.4f} / {acc_gui:.4f} | Lev: {lev_ar:.4f} / {lev_gui:.4f}\n\n"

final_acc_ar = acc_ar_total / total_chars
final_acc_gui = acc_gui_total / total_chars
final_lev_ar = lev_ar_total / len(songs)
final_lev_gui = lev_gui_total / len(songs)

avg_note_time_ar = total_time_ar / total_chars
avg_note_time_gui = total_time_gui / total_chars

output_log += f"TOTAL ACCURACY (AR): {final_acc_ar:.4f}\n"
output_log += f"TOTAL ACCURACY (GUI): {final_acc_gui:.4f}\n"
output_log += f"AVG NORM LEV (AR): {final_lev_ar:.4f}\n"
output_log += f"AVG NORM LEV (GUI): {final_lev_gui:.4f}\n"
output_log += f"AVG TIME PER NOTE (AR): {avg_note_time_ar:.6f}s\n"
output_log += f"AVG TIME PER NOTE (GUI): {avg_note_time_gui:.6f}s\n"

if total_train_time != 'N/A':
    output_log += f"TOTAL TRAIN TIME: {total_train_time:.2f}s\n"
    output_log += f"AVG EPOCH TIME: {avg_epoch_time:.4f}s\n"

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(output_log)

# ==========================================
# 5. GENERACIÓN DE GRÁFICOS (SUBPLOTS)
# ==========================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
x_axis = range(1, len(songs) + 1)

# Plot 1: Accuracy
ax1.plot(x_axis, song_acc_ar, marker='o', label='Acc Autorregresivo', color='tab:blue')
ax1.plot(x_axis, song_acc_gui, marker='s', label='Acc Guiado', color='tab:green')
ax1.axhline(final_acc_ar, color='tab:blue', linestyle='--', alpha=0.6)
ax1.axhline(final_acc_gui, color='tab:green', linestyle='--', alpha=0.6)
ax1.set_ylabel('Precisión (Accuracy)', fontweight='bold')
ax1.set_ylim(-0.05, 1.05)
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='lower right')
ax1.set_title('Comparativa de Métricas: Autorregresivo vs Guiado (LSTM)', fontweight='bold')

# Plot 2: Levenshtein
ax2.plot(x_axis, song_lev_ar, marker='o', label='Lev Autorregresivo', color='tab:orange')
ax2.plot(x_axis, song_lev_gui, marker='s', label='Lev Guiado', color='tab:purple')
ax2.axhline(final_lev_ar, color='tab:orange', linestyle='--', alpha=0.6)
ax2.axhline(final_lev_gui, color='tab:purple', linestyle='--', alpha=0.6)
ax2.set_xlabel('Índice de la Canción', fontweight='bold')
ax2.set_ylabel('Levenshtein Norm.', fontweight='bold')
ax2.set_ylim(-0.05, 1.05)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

# Guardado en CSV
file_exists = os.path.isfile(RESULTS_CSV)
with open(RESULTS_CSV, 'a', encoding='utf-8') as f:
    if not file_exists:
        f.write("hidden_size,lr,batch,acc_ar,acc_gui,lev_ar,lev_gui,time_note_ar,time_note_gui,total_train_time,avg_epoch_time\n")
    
    t_train_str = f"{total_train_time:.2f}" if isinstance(total_train_time, float) else str(total_train_time)
    t_epoch_str = f"{avg_epoch_time:.4f}" if isinstance(avg_epoch_time, float) else str(avg_epoch_time)
    
    f.write(f"{hidden_size},{learning_rate},{batch_size},{final_acc_ar:.4f},{final_acc_gui:.4f},{final_lev_ar:.4f},{final_lev_gui:.4f},{avg_note_time_ar:.6f},{avg_note_time_gui:.6f},{t_train_str},{t_epoch_str}\n")
"""
Script de inferencia y evaluación para el modelo Vanilla RNN (Multi-head).
Carga el modelo entrenado más reciente, genera secuencias musicales y
evalúa la precisión de la predicción respecto a las partituras originales.
"""

import os
import glob
import pickle
import numpy as np

# ==========================================
# 0. CONFIGURACIÓN DE RUTAS BASE
# ==========================================
DATASET_FILE = '../dataset/input.txt'
NAMES_FILE = '../dataset/song_names.txt'
OUTPUT_DIR = 'outputs'
RESULTS_CSV = 'results_rnn.csv'

# Asegurar que el directorio de outputs existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. CARGA DE DATOS Y DEL ÚLTIMO MODELO
# ==========================================
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    songs = [line.strip() for line in f if line.strip()]

with open(NAMES_FILE, 'r', encoding='utf-8') as f:
    song_names = [line.strip() for line in f if line.strip()]

def get_latest_model(models_dir='../models/', extension='rnn_*.pkl'):
    """Busca y retorna la ruta del modelo RNN con la fecha de modificación más reciente."""
    search_pattern = os.path.join(models_dir, extension)
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        raise FileNotFoundError(f"No se encontraron modelos con el patrón '{extension}' en '{models_dir}'.")
    return max(list_of_files, key=os.path.getmtime)

model_file = get_latest_model()
print(f"Cargando modelo: {model_file}")

with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Extracción de pesos e hiperparámetros
Wxh, Whh, bh = model['Wxh'], model['Whh'], model['bh']
Whys, bys = model['Whys'], model['bys'] 
char_to_ix = model['char_to_ix']
ix_to_char = model['ix_to_char']
hidden_size = model['hidden_size']
vocab_size = model['vocab_size']
seq_length = model['seq_length']
output_length = model['output_length']
learning_rate = model['learning_rate']
batch_size = model['batch_size']
epochs = model['epochs']

# Construcción dinámica del nombre del archivo de salida
lr_str = str(learning_rate).replace('.', '')
output_filename = f"rnn_{hidden_size}_{lr_str}_{batch_size}.txt"
output_path = os.path.join(OUTPUT_DIR, output_filename)

# ==========================================
# 2. FUNCIONES DE GENERACIÓN
# ==========================================
def generate(input_phrase, iterations):
    """
    Genera una secuencia de notas iterando sobre el bloque predictor multicabeza.
    
    Args:
        input_phrase (str): Secuencia de contexto inicial.
        iterations (int): Número de bloques de predicción (de tamaño output_length) a generar.
        
    Returns:
        str: Secuencia completa incluyendo el contexto inicial y lo generado.
    """
    current_phrase = input_phrase
    
    for _ in range(iterations):
        # 1. Procesamiento de las últimas notas de contexto para inicializar el estado oculto
        h_state = np.zeros((hidden_size, 1))
        context_notes = current_phrase[-seq_length:]
        
        for char in context_notes:
            if char in char_to_ix:
                ix = char_to_ix[char]
                x = np.zeros((vocab_size, 1))
                x[ix] = 1
                h_state = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h_state) + bh)
        
        # 2. Predicción simultánea a partir del estado oculto final
        new_chars = []
        for i in range(output_length):
            y = np.dot(Whys[i], h_state) + bys[i]
            p = np.exp(y) / np.sum(np.exp(y))
            
            # Selección codiciosa (Greedy)
            ix_pred = np.argmax(p.ravel())
            new_chars.append(ix_to_char[ix_pred])
            
        current_phrase += "".join(new_chars)
        
    return current_phrase

# ==========================================
# 3. EVALUACIÓN Y EXPORTACIÓN DE RESULTADOS
# ==========================================
input_size = 12 
output_log = ''
accuracy_total = 0
total_chars_predicted = 0

print("\nGenerando predicciones y calculando métricas...\n")

for i in range(len(songs)):
    song = songs[i]
    song_name = song_names[i]
    
    input_phrase = song[:input_size]
    
    # Cálculo de iteraciones necesarias para igualar la longitud de la canción original
    notes_to_generate = len(song) - input_size
    iterations_needed = int(np.ceil(notes_to_generate / output_length))
    
    pred_song = generate(input_phrase, iterations_needed)
    
    # Recorte de la predicción excedente del último bloque
    pred_song = pred_song[:len(song)]
    
    # Evaluación de precisión por carácter
    matched_notes = [song[j] == pred_song[j] for j in range(input_size, len(song))]
    correct_matches = sum(matched_notes)
    accuracy_song = correct_matches / notes_to_generate
    
    accuracy_total += correct_matches
    total_chars_predicted += notes_to_generate
    
    output_log += f"{song_name}\n"
    output_log += f"Expected:  {song}\n"
    output_log += f"Predicted: {pred_song}\n"
    output_log += f"Accuracy:  {accuracy_song:.4f}\n\n"

# Precisión global
final_accuracy = accuracy_total / total_chars_predicted
output_log += f"TOTAL ACCURACY: {final_accuracy:.4f}\n"

print(output_log)

# Guardado del log de texto detallado
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(output_log)

# Guardado de métricas estructuradas en CSV
file_exists = os.path.isfile(RESULTS_CSV)

with open(RESULTS_CSV, 'a', encoding='utf-8') as f:
    if not file_exists:
        # Se escribe la cabecera si el archivo se acaba de crear
        f.write("hidden_size,seq_length,learning_rate,batch_size,epochs,input_size,accuracy\n")
    
    f.write(f"{hidden_size},{seq_length},{learning_rate},{batch_size},{epochs},{input_size},{final_accuracy:.4f}\n")

print(f"Resultados guardados en '{output_path}' y métricas en '{RESULTS_CSV}'.")
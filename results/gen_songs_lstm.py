"""
Script de inferencia y evaluación para el modelo LSTM (Encoder-Decoder).
Carga el modelo entrenado más reciente, genera secuencias musicales y
evalúa la precisión de la predicción respecto a las partituras originales.
"""

import os
import glob
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Ocultar mensajes de información de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 0. CONFIGURACIÓN DE RUTAS BASE
# ==========================================
DATASET_FILE = '../dataset/input.txt'
NAMES_FILE = '../dataset/song_names.txt'
OUTPUT_DIR = 'outputs'
RESULTS_CSV = 'results_lstm.csv'

# Asegurar que el directorio de outputs existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 1. CARGA DE DATOS Y METADATOS
# ==========================================
with open(DATASET_FILE, 'r', encoding='utf-8') as f:
    songs = [line.strip() for line in f if line.strip()]

with open(NAMES_FILE, 'r', encoding='utf-8') as f:
    song_names = [line.strip() for line in f if line.strip()]

def get_latest_metadata(models_dir='../models/', extension='lstm_*.pkl'):
    """Busca y retorna la ruta del archivo de metadatos LSTM más reciente."""
    search_pattern = os.path.join(models_dir, extension)
    list_of_files = glob.glob(search_pattern)
    if not list_of_files:
        raise FileNotFoundError(f"No se encontraron metadatos con el patrón '{extension}' en '{models_dir}'.")
    return max(list_of_files, key=os.path.getmtime)

meta_file = get_latest_metadata()
print(f"Cargando metadatos: {meta_file}")

with open(meta_file, 'rb') as f:
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

# Construcción dinámica del nombre del archivo de salida
lr_str = str(learning_rate).replace('.', '')
output_filename = f"lstm_{hidden_size}_{lr_str}_{batch_size}.txt"
output_path = os.path.join(OUTPUT_DIR, output_filename)

# ==========================================
# 2. CARGA DEL MODELO KERAS
# ==========================================
# Se asume que el modelo .keras tiene exactamente el mismo prefijo que el .pkl
model_keras_file = meta_file.replace('.pkl', '.keras')

if not os.path.exists(model_keras_file):
    raise FileNotFoundError(f"No se encontró el modelo Keras asociado: {model_keras_file}")

print(f"Cargando pesos Keras: {model_keras_file}")
model = load_model(model_keras_file)

# ==========================================
# 3. FUNCIONES DE GENERACIÓN
# ==========================================
def generate_song(model, seed_phrase, generate_size):
    """
    Genera una secuencia de notas iterando sobre el bloque predictor Encoder-Decoder.
    
    Args:
        model: Modelo LSTM Keras entrenado.
        seed_phrase (str): Secuencia de contexto inicial.
        generate_size (int): Cantidad de notas totales a predecir.
        
    Returns:
        str: Secuencia completa incluyendo el contexto inicial y lo generado.
    """
    resultado = seed_phrase
    
    # Cálculo de bloques necesarios a generar
    iterations = int(np.ceil(generate_size / pred_length))
    
    for _ in range(iterations):
        seq_in = resultado[-seq_length:]
        x_pred = [char_to_ix[c] for c in seq_in]
        x_pred = to_categorical([x_pred], num_classes=vocab_size)
        
        # predict devuelve una matriz tridimensional; [0] extrae el batch
        pred_probs = model.predict(x_pred, verbose=0)[0] 
        
        # Decodificación de las 4 notas simultáneamente
        new_chars = []
        for i in range(pred_length):
            ix_pred = np.argmax(pred_probs[i]) # Selección codiciosa (Greedy)
            new_chars.append(ix_to_char[ix_pred])
            
        resultado += "".join(new_chars)
        
    return resultado

# ==========================================
# 4. EVALUACIÓN Y EXPORTACIÓN DE RESULTADOS
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
    size_to_generate = len(song) - input_size
    
    pred_song = generate_song(model, input_phrase, size_to_generate)
    
    # Recorte por exceso de generación en el último bloque multicabeza
    pred_song = pred_song[:len(song)]
    
    matched_notes = [song[j] == pred_song[j] for j in range(input_size, len(song))]
    correct_matches = sum(matched_notes)
    accuracy_song = correct_matches / size_to_generate
    
    accuracy_total += correct_matches
    total_chars_predicted += size_to_generate
    
    output_log += f"{song_name}\n"
    output_log += f"Expected:  {song}\n"
    output_log += f"Predicted: {pred_song}\n"
    output_log += f"Accuracy:  {accuracy_song:.4f}\n\n"

# Cálculo de precisión global ponderada
final_accuracy = accuracy_total / total_chars_predicted
output_log += f"TOTAL ACCURACY: {final_accuracy:.4f}\n"

print(output_log)

# Guardado del log textual
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(output_log)

# Guardado estructurado de métricas en CSV
file_exists = os.path.isfile(RESULTS_CSV)

with open(RESULTS_CSV, 'a', encoding='utf-8') as f:
    if not file_exists:
        # Se escribe la cabecera si el archivo se acaba de crear
        f.write("hidden_size,seq_length,learning_rate,batch_size,epochs,input_size,accuracy\n")
    
    f.write(f"{hidden_size},{seq_length},{learning_rate},{batch_size},{epochs},{input_size},{final_accuracy:.4f}\n")

print(f"Resultados guardados en '{output_path}' y métricas en '{RESULTS_CSV}'.")
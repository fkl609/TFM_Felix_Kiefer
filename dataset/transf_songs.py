"""
Preprocesamiento de partituras XML para extracción de secuencias.

Este script analiza un directorio de archivos en formato MusicXML, extrae 
la información de notas musicales y octavas, y genera una representación 
simbólica unidimensional (un carácter por nota). Esta transformación 
es necesaria para construir el dataset ('input.txt') que 
alimentará los modelos RNN y LSTM.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np

def parse_song_notes(root):
    """
    Extrae notas y octavas de un archivo XML y las mapea a caracteres.

    Args:
        root (xml.etree.ElementTree.Element): Raíz del árbol XML parseado.

    Returns:
        str: Secuencia continua de caracteres representando la melodía.
             Devuelve una cadena vacía si no se encuentran notas.
    """
    notes_raw = []
    octaves = []

    for part in root.findall('part'):
        for measure in part.findall('measure'):
            for note in measure.findall('note'):
                pitch = note.find('pitch')
                if pitch is not None:
                    step = pitch.find('step').text
                    octave = int(pitch.find('octave').text)
                    notes_raw.append(step)
                    octaves.append(octave)
    
    if not notes_raw:
        return ""
        
    notes_raw = np.array(notes_raw)
    octaves = np.array(octaves)
    
    # Normalizar octavas para que la mínima de la canción sea 0
    octaves -= np.min(octaves)
    
    # Diccionarios de mapeo
    notas_o0 = 'abcdefg'
    notas_o1 = notas_o0.upper()
    notas_o2 = 'hijklmn'
    notas_o3 = notas_o2.upper()
    notas_octavas = [notas_o0, notas_o1, notas_o2, notas_o3]

    # Mapeo eficiente
    sorter = np.argsort(list(notas_o1))
    ix_notas = sorter[np.searchsorted(list(notas_o1), notes_raw, sorter=sorter)]

    notas = "".join([notas_octavas[octaves[i]][ix_notas[i]] for i in range(len(notes_raw))])
    return notas

def process_folder(input_folder, output_file, names_file):
    """
    Procesa todos los XML de una carpeta y exporta las secuencias y nombres.
    """
    if not os.path.exists(input_folder):
        print(f"Error: No se encontró el directorio '{input_folder}'.")
        return

    all_songs = []
    song_names = []

    for file in sorted(os.listdir(input_folder)):
        if file.endswith('.xml'):
            print(f"Procesando: {file}")
            path = os.path.join(input_folder, file)
            
            try:
                tree = ET.parse(path)
                notes_sequence = parse_song_notes(tree.getroot())
                
                if notes_sequence:
                    all_songs.append(notes_sequence)
                    song_names.append(file[:-4])
            except ET.ParseError:
                print(f"Error al leer {file}. Omitiendo...")

    if all_songs:
        # Asegurar que el directorio de salida del txt de canciones exista (si hay)
        out_dir_songs = os.path.dirname(output_file)
        if out_dir_songs:
            os.makedirs(out_dir_songs, exist_ok=True)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(all_songs) + "\n")
        
        # Asegurar que el directorio de salida de los nombres exista (si hay)
        out_dir_names = os.path.dirname(names_file)
        if out_dir_names:
            os.makedirs(out_dir_names, exist_ok=True)
            
        with open(names_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(song_names) + "\n")
            
        print(f"\nÉxito: {len(all_songs)} canciones extraídas.")
        print(f"Secuencias -> '{output_file}'")
        print(f"Nombres    -> '{names_file}'")
    else:
        print("No se encontraron secuencias válidas.")

if __name__ == '__main__':
    FOLDER_PATH = './xml_songs'
    OUTPUT_TEXT = 'input.txt'
    OUTPUT_NAMES = 'song_names.txt'
    
    process_folder(FOLDER_PATH, OUTPUT_TEXT, OUTPUT_NAMES)
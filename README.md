# TFM: Predicción y Generación de Melodías con Arquitecturas RNN y LSTM

Este repositorio contiene el código fuente asociado al Trabajo de Fin de Máster (Máster en Ciencia de Datos, Universidad de Cantabria). 

El proyecto aborda el modelado secuencial simbólico para aprender y reproducir melodías (notación ABC) mediante redes neuronales recurrentes. Se compara una arquitectura Vanilla RNN frente a una LSTM (Encoder-Decoder) utilizando un enfoque de predicción simultánea multicabeza.

## Estructura del Proyecto

* **`dataset/`**: Archivos de origen XML (canciones infantiles) y script (`transf_songs.py`) para la extracción de las notas en secuencias de texto continuo (`input.txt`).
* **`model_scripts/`**: Código fuente para la optimización de hiperparámetros (Optuna y Keras Tuner) y el entrenamiento de los modelos (`rnn.py` y  `lstm.py`).
* **`models/`**: Directorio de almacenamiento para los pesos guardados (`.keras`) y diccionarios de metadatos (`.pkl`).
* **`results/`**: Scripts de inferencia (`gen_songs_rnn.py`, `gen_songs_lstm.py`). Incluye las carpetas `outputs/` (métricas por carácter y secuencias generadas en `.txt` y `.csv`) y `plots/` (curvas de pérdida).

## Entorno y Requisitos

Se ofrecen dos alternativas para replicar el entorno virtual del proyecto:

**Opción A: Usando Conda (Recomendado - Miniforge/Anaconda)**
```bash
conda env create -f environment.yml
conda activate tfm
```

**Opción B: Usando pip**
```bash
python -m venv tfm_env
tfm_env\Scripts\activate # En Linux/macOS utilizar source tfm_env/bin/activate
pip install -r requirements.txt
```

## Flujo de Ejecución (Pipeline)

1. **Preprocesamiento:** Ejecutar `transf_songs.py` dentro de la carpeta `dataset/` para transformar los XML en el corpus de entrenamiento `input.txt` y extraer `song_names.txt`. *(Opcional, ya que los archivos ya han sido generados)*
2. **Entrenamiento y Optimización:** Ejecutar los scripts en `model_scripts/`. Estos realizarán la búsqueda de hiperparámetros, entrenarán el modelo óptimo y guardarán los artefactos en `models/` y las curvas en `results/plots/`.
3. **Inferencia y Evaluación:** Ejecutar los scripts de la carpeta `results/`. Buscarán automáticamente el último modelo entrenado, generarán las melodías iterando sobre el bloque predictor y exportarán las métricas de precisión a `results/outputs/`.
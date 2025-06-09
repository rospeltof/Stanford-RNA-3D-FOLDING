# Stanford-RNA 3D-FOLDING
 Solución para competencia de stanford RNA 3d folding

Este repositorio contiene todo el pipeline para el análisis de datos, modelado y evaluación de predicción de estructuras tridimensionales de ARN usando Deep Learning y embeddings de RNABERT.

## Estructura del proyecto
```bash
code/
  01_EDA.ipynb
  02_Data_Processing.ipynb
  03_Modeling.ipynb
  04_Nuevos_Enfoques_Datos.ipynb
  metric_utils.py

data/
  MSA/
  labeled/
  processed/
    entropy_data_actualizado.csv
  sample_submission.csv
  train_sequences.csv
  train_labels.csv
  validation_sequences.csv
  validation_labels.csv

.gitignore
.gitattributes
README.md
environment.yml # Archivo con la lista de librerías y versiones necesarias
```
## Descripción de los notebooks

1. **01_EDA.ipynb**  
   - Estadísticas básicas de las secuencias  
   - Distribución de longitudes, composición de nucleótidos  
   - Inspección de coordenadas y filtrado de outliers  

2. **02_Data_Processing.ipynb**  
   - Limpieza de archivos originales  
   - Cálculo de entropía por posición a partir de los FASTA de MSA  
   - Creación de features: posición relativa, entropía transformada, codificación one-hot de nucleótidos  
   - Guardado de los datos procesados en `data/processed/`

3. **03_Modeling.ipynb**  
   - Baseline con red feed-forward (sin secuencias)  
   - Modelo CNN 1D para predicción de coordenadas C1’  
   - Experimentos con funciones de pérdida (MSE, MAE, MAPE, Huber, etc.)  
   - Trazado de curvas de pérdida y TM-score usando `metric_utils.py`

4. **04_Nuevos_Enfoques_Datos.ipynb**  
   - Integración de embeddings de RNABERT para enriquecer las entradas  
   - Ajuste y evaluación del modelo CNN + embeddings  
   - Arquitectura “Pequeño Transformer” y comparación de resultados  
   - Inferencia final sobre 12 secuencias de prueba y cómputo de TM-score promedio  

## Uso de `metric_utils.py`

Este script contiene:

- **compute_tm_score()**: función para calcular TM-score entre coordenadas verdaderas y predichas.  
- **TMScoreCallback**: callback de Keras que evalúa y grafica TM-score en cada época.  
- Funciones de visualización (p. ej. trazado de curvas de pérdida y TM-score).

## Descripción de entropy_data_actualizado.csv
Archivo tabular en el que cada columna corresponde a una de las secuencias disponibles y cada fila (fila i) representa la entropía en la posición i del nucleótido para cada secuencia.

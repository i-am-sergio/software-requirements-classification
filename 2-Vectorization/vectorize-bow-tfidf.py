from math import log
import numpy as np
import pandas as pd

# Leer el archivo CSV normalizado de la etapa previa
df = pd.read_csv(
    '../1-Preprocessing/output/dataset_normalized.csv',  # Ruta al archivo de entrada
    sep=',',  # Separador de columnas
    header=0,  # Indica que la primera fila contiene los encabezados
    quotechar='"',  # Carácter para manejar comillas
    doublequote=True  # Permitir comillas dobles dentro de campos
)

# Paso 1: Crear un conjunto de características únicas y una lista de vectores para las filas
features_set = set()  # Conjunto para almacenar características únicas
vector_list = [None] * df.shape[0]  # Lista para almacenar vectores por fila

for index, row in df.iterrows():
    text = row['RequirementText']  # Texto de la fila actual
    tokens = text.split(' ')  # Tokenizar el texto (dividir por espacios)
    row_vector = {}  # Diccionario para almacenar el conteo de cada token en la fila
    
    for token in tokens:
        features_set.add(token)  # Agregar el token al conjunto de características
        row_vector[token] = row_vector.get(token, 0) + 1  # Incrementar el conteo del token
    
    vector_list[index] = row_vector  # Guardar el vector de la fila actual

# Ordenar las características alfabéticamente para mantener consistencia
features_sorted = sorted(features_set)
# Crear un mapa de búsqueda para acceder rápidamente al índice de cada característica
features_sorted_lookup_map = {value: index for index, value in enumerate(features_sorted)}


# ====== BoW (Bag of Words) ======
# Paso 2: Construir la matriz BoW como una matriz densa
vector_matrix_bow = np.zeros((df.shape[0], len(features_set)), dtype=np.int16)  # Inicializar la matriz

for i, row_dict in enumerate(vector_list):
    for token, count in row_dict.items():
        vector_matrix_bow[i, features_sorted_lookup_map[token]] = count  # Asignar el conteo en la posición correspondiente

# Convertir la matriz BoW en un DataFrame y guardar en un archivo CSV
bow_df = pd.DataFrame(data=vector_matrix_bow, columns=features_sorted)
bow_df['_class_'] = df['_class_']  # Agregar la columna de clases
bow_df.to_csv('./output/dataset_bow.csv', sep=',', header=True, index=False)


# ====== TF-IDF (Term Frequency-Inverse Document Frequency) ======
# Paso 3: Construir la matriz TF-IDF
vector_matrix_tfidf = np.zeros((df.shape[0], len(features_set)))  # Inicializar la matriz TF-IDF
row_sums_bow = vector_matrix_bow.sum(axis=1)  # Suma de palabras por fila
col_sums_bow = vector_matrix_bow.sum(axis=0)  # Suma de palabras por columna

# Calcular el TF-IDF para cada posición de la matriz
for i in range(vector_matrix_bow.shape[0]):
    for j in range(vector_matrix_bow.shape[1]):
        if vector_matrix_bow[i, j] == 0:  # Evitar cálculos innecesarios
            continue
        # Calcular Term Frequency (TF)
        tf = vector_matrix_bow[i, j] / row_sums_bow[i]
        # Calcular Inverse Document Frequency (IDF)
        idf = log(vector_matrix_bow.shape[0] / col_sums_bow[j])
        # Asignar el valor TF-IDF
        vector_matrix_tfidf[i, j] = tf * idf

# Convertir la matriz TF-IDF en un DataFrame y guardar en un archivo CSV
tfidf_df = pd.DataFrame(data=vector_matrix_tfidf, columns=features_sorted)
tfidf_df['_class_'] = df['_class_']  # Agregar la columna de clases
tfidf_df.to_csv('./output/dataset_tfidf.csv', sep=',', header=True, index=False)

# Impresion de resumen
print("===== Vectorizacion completada =====")
print(f"Matrices generadas: BoW -> {vector_matrix_bow.shape}, TF-IDF -> {vector_matrix_tfidf.shape}")
print("Archivos guardados en './output/dataset_bow.csv' y './output/dataset_tfidf.csv'")


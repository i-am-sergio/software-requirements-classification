from math import log # natural logarithm
import numpy as np 
import pandas as pd

# Lee el dataset normalizado
df = pd.read_csv('../2-preprocessing/output/dataset_normalized.csv', sep=',', header=0, quotechar = '"', doublequote=True)

# manual vectorization; not using CountVectorizer and TfidfVectorizer from sklearn.feature_extraction

# crea un conjunto de características y una lista de vectores
features_set = set()
vector_list = [None] * df.shape[0]

# crea un vector por cada fila del dataset
for index, row in df.iterrows():
    text = row['RequirementText'] # texto de la fila
    tokens = text.split(' ') # tokeniza el texto
    row_vector = {} # vector de la fila
    for token in tokens: # por cada token
        features_set.add(token) # agrega el token al conjunto de características
        if token in row_vector: # si el token ya está en el vector de la fila
            row_vector[token] = row_vector[token] + 1 # incrementa la frecuencia del token
        else:   # si el token no está en el vector de la fila
            row_vector[token] = 1 # agrega el token al vector de la fila con frecuencia 1
    vector_list[index] = row_vector # agrega el vector de la fila a la lista de vectores

features_sorted = list(features_set) # convierte el conjunto de características en una lista
features_sorted.sort() # ordena la lista de características
features_sorted_lookup_map = { value: index for index, value in enumerate(features_sorted) } # crea un mapa de búsqueda de características

vector_matrix_bow = np.zeros((df.shape[0], len(features_set)), dtype=np.int16) # crea una matriz de ceros de tamaño (filas del dataset, características)

# llena la matriz de vectores con los vectores de la lista de vectores
for i in range(len(vector_list)): # por cada vector de la lista de vectores
    row_dict = vector_list[i]
    for k, v in row_dict.items(): # por cada par (clave, valor) del vector
        vector_matrix_bow[i, features_sorted_lookup_map[k]] = v # asigna el valor del vector a la matriz de vectores

# --- BOW ---
# guarda la matriz de vectores en un archivo CSV
bow_df = pd.DataFrame(data=vector_matrix_bow, columns=features_sorted)
bow_df['_class_'] = df['_class_'] # agrega la columna de clases
bow_df.to_csv('./output/dataset_bow.csv', sep=',', header=True, index=False) # guarda la matriz de vectores en un archivo CSV

# --- TF-IDF ---
# crea una matriz de vectores TF-IDF
vector_matrix_tfidf = np.zeros((df.shape[0], len(features_set)))
row_sums_bow = vector_matrix_bow.sum(axis=1) 
col_sums_bow = vector_matrix_bow.sum(axis=0)

# calcula el valor TF-IDF de cada celda de la matriz de vectores
for i in range(vector_matrix_bow.shape[0]):
    for j in range(vector_matrix_bow.shape[1]):
        if vector_matrix_bow[i, j] == 0:    # minor optimization
            continue
        tf = vector_matrix_bow[i, j] / row_sums_bow[i]
        idf = log(vector_matrix_bow.shape[0] / col_sums_bow[j])
        vector_matrix_tfidf[i, j] = tf * idf

# guarda la matriz de vectores TF-IDF en un archivo CSV
tfidf_df = pd.DataFrame(data=vector_matrix_tfidf, columns=features_sorted)
tfidf_df['_class_'] = df['_class_']
tfidf_df.to_csv('./output/dataset_tfidf.csv', sep=',', header=True, index=False)

# bow_df and tfidf_df are (very) sparse matrices of shape (969, 1524)

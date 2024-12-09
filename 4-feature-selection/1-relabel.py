import pandas as pd

import os, sys
sys.path.insert(0, os.path.abspath('..'))
from utils import utils

bow_df = pd.read_csv('../3-vectorization/output/dataset_bow.csv', sep=',', header=0)
tfidf_df = pd.read_csv('../3-vectorization/output/dataset_tfidf.csv', sep=',', header=0)

# *** Creación de datasets con 12 clases ***
# No se realizan cambios en el conjunto, simplemente se renombran para mantener consistencia
bow_twelve_df = bow_df.copy()  # Copia del DataFrame BoW
bow_twelve_df.to_csv('./output/1-bow-12.csv', sep=',', header=True, index=False)  # Guardar en CSV

tfidf_twelve_df = tfidf_df.copy()  # Copia del DataFrame TF-IDF
tfidf_twelve_df.to_csv('./output/1-tfidf-12.csv', sep=',', header=True, index=False)  # Guardar en CSV

# *** Creación de datasets con 11 clases ***
# Eliminar registros donde la clase sea 'F'
bow_eleven_df = bow_df[bow_df['_class_'] != 'F'].copy()  # Filtrar registros y copiar
bow_eleven_df.to_csv('./output/1-bow-11.csv', sep=',', header=True, index=False)  # Guardar en CSV

tfidf_eleven_df = tfidf_df[tfidf_df['_class_'] != 'F'].copy()  # Filtrar registros y copiar
tfidf_eleven_df.to_csv('./output/1-tfidf-11.csv', sep=',', header=True, index=False)  # Guardar en CSV

# *** Creación de datasets con 2 clases ***
# Definir una función lambda para binarizar las clases (F: Funcional, NF: No funcional)
binarize_class = lambda entry: 'F' if entry == 'F' else 'NF'

# Aplicar binarización a las clases en el dataset BoW
bow_two_df = bow_df.copy()
bow_two_df['_class_'] = bow_two_df['_class_'].apply(utils.binarize_class_variable)  # Binarizar usando utils
bow_two_df.to_csv('./output/1-bow-2.csv', sep=',', header=True, index=False)  # Guardar en CSV

# Aplicar binarización a las clases en el dataset TF-IDF
tfidf_two_df = tfidf_df.copy()
tfidf_two_df['_class_'] = tfidf_two_df['_class_'].apply(utils.binarize_class_variable)  # Binarizar usando utils
tfidf_two_df.to_csv('./output/1-tfidf-2.csv', sep=',', header=True, index=False)  # Guardar en CSV

# *** Resumen ***
print("Archivos generados:")
print("- Dataset BoW (12 clases): ./output/1-bow-12.csv")
print("- Dataset TF-IDF (12 clases): ./output/1-tfidf-12.csv")
print("- Dataset BoW (11 clases): ./output/1-bow-11.csv")
print("- Dataset TF-IDF (11 clases): ./output/1-tfidf-11.csv")
print("- Dataset BoW (2 clases): ./output/1-bow-2.csv")
print("- Dataset TF-IDF (2 clases): ./output/1-tfidf-2.csv")
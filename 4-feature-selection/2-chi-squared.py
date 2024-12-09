import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

def process_dataset(name, stat_significance_level):
    """
    Procesa un dataset para reducir sus características utilizando la prueba de Chi-cuadrado.
    
    Args:
        name (str): Nombre del archivo base (sin ruta).
        stat_significance_level (float): Nivel de significancia estadística para seleccionar características.

    Returns:
        tuple: Una tupla que contiene el número de características seleccionadas y los nombres de las 10 mejores características.
    """
    # Leer el dataset
    df = pd.read_csv(f'./output/1-{name}.csv', sep=',', header=0)

    # Separar las características (X) y las etiquetas de clase (y)
    X = df.drop('_class_', axis=1)  # Eliminar la columna '_class_'
    y = df['_class_']  # Etiquetas

    # Selección de características con Chi-cuadrado
    selection = SelectKBest(chi2, k='all').fit(X, y)
    features_criterion = selection.pvalues_ < stat_significance_level

    # Crear un nuevo dataset con las características seleccionadas
    features_criterion_select = np.append(arr=features_criterion, values=[True], axis=0)  # Incluir '_class_'
    output_df = df.loc[:, features_criterion_select]
    output_df.to_csv(f'./output/2-{name}.csv', sep=',', header=True, index=False)  # Guardar el dataset reducido

    # Mapear características a sus puntuaciones y seleccionar las 10 mejores
    features_scores_map = {X.columns[index]: selection.scores_[index] for index in range(len(X.columns))}
    top_ten_feature_names = sorted(features_scores_map, key=features_scores_map.get, reverse=True)[:10]

    return len(output_df.columns) - 1, top_ten_feature_names

# Listas de datasets y niveles de significancia estadística
datasets = ['bow-12', 'bow-11', 'bow-2', 'tfidf-12', 'tfidf-11', 'tfidf-2']
stat_significances = [0.075, 0.075, 0.075, 0.8, 0.8, 0.325]  # Basados en el rendimiento del modelo MNB

# Procesar cada dataset y recopilar información
infos = [process_dataset(dataset, stat_significances[index]) for index, dataset in enumerate(datasets)]

# Crear líneas con el resumen de características y las 10 mejores para cada dataset
lines = [
    f"** {datasets[index]} **\ncount: {info[0]}\ntop ten: {', '.join(info[1])}\n\n"
    for index, info in enumerate(infos)
]
lines[-1] = lines[-1][:-1]  # Eliminar el salto de línea extra al final

# Guardar la información en un archivo de texto
with open('./output/2-chi-sq-info.txt', 'w') as writer:
    writer.writelines(lines)

# *** Resumen ***
print("Archivos generados:")
print("- Datasets reducidos: ./output/2-<dataset>.csv")
print("- Información Chi-cuadrado: ./output/2-chi-sq-info.txt")

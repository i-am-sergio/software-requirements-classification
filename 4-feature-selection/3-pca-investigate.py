import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

# Lista de datasets a analizar
datasets = ['bow-12', 'bow-11', 'bow-2', 'tfidf-12', 'tfidf-11', 'tfidf-2']

def investigate_dataset(name, axs):
    """
    Analiza un dataset utilizando PCA y grafica la suma acumulada de la varianza explicada.
    
    Args:
        name (str): Nombre del dataset (sin ruta).
        axs (matplotlib.axes._subplots.AxesSubplot): Subgráfico donde se graficará la curva.
    """
    # Cargar el dataset
    df = pd.read_csv(f'./output/2-{name}.csv', sep=',', header=0)
    X = df.drop('_class_', axis=1)  # Eliminar la columna '_class_'

    # Calcular el número de componentes posibles (mínimo entre filas y columnas)
    components_count = min(X.shape[0], X.shape[1])

    # Aplicar PCA con todos los componentes posibles
    pca = PCA(n_components=components_count)
    pca.fit(X)

    # Calcular la suma acumulada de la varianza explicada
    explained_variance_cumsum = pd.Series(data=pca.explained_variance_ratio_).sort_values(ascending=False).cumsum()

    # Configurar y graficar los resultados en el subgráfico
    axs.set_title(name)
    axs.plot(explained_variance_cumsum)
    axs.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())  # Subdivisiones menores en el eje X
    axs.set_yticks(np.linspace(0, 1, 11), minor=False)  # Etiquetas mayores en el eje Y
    axs.set_yticks(np.linspace(0, 1, 21), minor=True)  # Etiquetas menores en el eje Y
    axs.grid(color='#e0e0e0', which='major')  # Configurar rejilla para etiquetas mayores
    axs.grid(color='#f2f2f2', which='minor')  # Configurar rejilla para etiquetas menores

# Crear una figura con una cuadrícula de subgráficos (2x3)
fig, axs = plt.subplots(2, 3)
fig.suptitle('PCA Explained Variance Cumulative Sums')  # Título general
fig.set_size_inches(16, 12)  # Tamaño de la figura

# Iterar sobre los datasets y asignar cada uno a un subgráfico
for index, dataset in enumerate(datasets):
    investigate_dataset(dataset, axs[math.floor(index / 3), index % 3])  # Ubicación en la cuadrícula

# Guardar la figura generada
fig.savefig('./output/3-pca-investigation.png')

# *** Resumen ***
print("Análisis PCA completado. Gráfico guardado en './output/3-pca-investigation.png'")

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Lista de datasets y el número de componentes principales a conservar para cada uno
datasets = ['bow-12', 'bow-11', 'bow-2', 'tfidf-12', 'tfidf-11', 'tfidf-2']
component_counts = [300, 200, 250, 400, 300, 500]  # Basado en las curvas de suma acumulada de varianza explicada (98%-99%)

# Imprimir encabezado para mostrar la varianza explicada
print('** Varianza Explicada **')

def transform_dataset(name, component_count):
    """
    Aplica PCA a un dataset y transforma sus características reduciendo su dimensionalidad.

    Args:
        name (str): Nombre del dataset (sin ruta).
        component_count (int): Número de componentes principales a conservar.
    """
    # Cargar el dataset
    df = pd.read_csv(f'./output/2-{name}.csv', sep=',', header=0)
    X = df.drop('_class_', axis=1)  # Eliminar la columna '_class_' (variable objetivo)

    # Aplicar PCA con el número de componentes principales especificado
    pca = PCA(n_components=component_count)
    pca.fit(X)

    # Imprimir el porcentaje de varianza explicada por los componentes seleccionados
    explained_variance = sum(pca.explained_variance_ratio_[:component_count]) * 100
    print(f'{name:<9}: {explained_variance:.3f}%')

    # Transformar los datos originales al nuevo espacio reducido
    X_new = pca.transform(X)

    # Crear un nuevo DataFrame con los componentes principales y la clase
    X_new_cols = [f'Comp{index + 1}' for index in range(X_new.shape[1])]
    df_output = pd.DataFrame(data=X_new, columns=X_new_cols)
    df_output['_class_'] = df['_class_']

    # Guardar el dataset transformado en un archivo CSV
    df_output.to_csv(f'./output/3-{name}.csv', sep=',', header=True, index=False)

# Iterar sobre los datasets y transformarlos
for index, dataset in enumerate(datasets):
    transform_dataset(dataset, component_counts[index])

# *** Resumen ***
print("\nTransformación completada. Archivos generados en './output/3-<nombre_dataset>.csv'.")

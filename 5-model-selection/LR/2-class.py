import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os
import sys

# Agregar la ruta del módulo de utilidades
sys.path.insert(0, os.path.abspath("../.."))
from utils import utils

# Definir las clases objetivo
class_names = ['F', 'NF']
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

def model_select(dataset):
    """
    Realiza selección de modelo utilizando validación cruzada con k-folds
    y evalúa diferentes valores para el hiperparámetro 'C' de la regresión logística.

    Args:
        dataset (str): Nombre del dataset (sin extensión ni ruta).

    Returns:
        str: Información de la evaluación de los modelos, incluyendo métricas para cada hiperparámetro.
    """
    output_info = f'** {dataset} **\n'
    
    # Cargar el dataset utilizando una función personalizada de utils
    df = utils.load_dataset('../../4-feature-selection/output', dataset, True)

    print(df.head(10))
    print(df.shape)
    print(df.dtypes)

    # Configuración para validación cruzada
    K = 10  # Número de folds
    hyperparam_candidates = [0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 1000]
    i = 0  # Contador de pasos

    # Iterar sobre los valores candidatos del hiperparámetro 'C'
    for hyperparam_candidate in hyperparam_candidates:
        random_split = utils.cv_split(df, K)  # Dividir el dataset en K folds
        current_f1 = 0  # Métrica acumulada para el hiperparámetro actual

        # Realizar validación cruzada
        for j in range(K):
            test = random_split[j]
            training_list = random_split[0:j] + random_split[j+1:K]
            training = pd.concat(training_list)

            # Separar características y etiquetas para entrenamiento y prueba
            X_train = training.drop('_class_', axis=1)
            Y_train = label_encoder.transform(training['_class_'])
            X_test = test.drop('_class_', axis=1)
            Y_test = label_encoder.transform(test['_class_'])

            # Entrenar modelo con el hiperparámetro candidato
            model = LogisticRegression(C=hyperparam_candidate, class_weight='balanced')
            model.fit(X_train, Y_train)

            # Evaluar el modelo
            results = utils.estimate_model_performance(model, X_test, Y_test)
            current_f1 += results[0]  # Acumular F1-score

        # Calcular el promedio de F1-score para el hiperparámetro actual
        current_f1 /= K
        output_info += f'step {i + 1}: {hyperparam_candidate} - {current_f1}\n'
        i += 1

    return output_info

# Seleccionar modelo para los datasets bow-2 y tfidf-2
bow = model_select('bow-2')
tfidf = model_select('tfidf-2')

# Imprimir resultados
print(f'{bow}\n{tfidf}')

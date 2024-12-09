import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os
import sys

# Agregar la ruta del módulo de utilidades
sys.path.insert(0, os.path.abspath("../.."))
from utils import utils

# Configuración de validación cruzada
K = 10  # Número de folds para validación cruzada
SELECTED_C = 1  # Valor del hiperparámetro 'C' seleccionado para la regresión logística

# Definir las clases objetivo
class_names = ['F', 'NF']
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

# Variable global para almacenar el modelo entrenado
trained_model = None

def evaluate_binary_model(dataset, output_filename):
    """
    Evalúa un modelo binario con validación cruzada k-fold y escribe los resultados
    en un archivo de texto. También calcula métricas promedio en los conjuntos de
    entrenamiento y prueba.

    Args:
        dataset (str): Nombre del dataset (sin extensión ni ruta).
        output_filename (str): Nombre del archivo donde se almacenarán los resultados.

    Returns:
        str: Resumen de las métricas obtenidas durante la evaluación.
    """
    global trained_model
    output_info = f'** {dataset} **'
    output = open(f'../results/{output_filename}.txt', "w")

    # Variables para acumular resultados
    training_results = [0, 0, 0, 0]  # [precisión, recall, f1_score, exactitud] en entrenamiento
    test_results = [0, 0, 0, 0]  # [precisión, recall, f1_score, exactitud] en prueba

    # Cargar el dataset
    df = utils.load_dataset('../../4-feature-selection/output', dataset, True)
    random_split = utils.cv_split(df, K)  # Dividir el dataset en K folds

    # Validación cruzada
    for j in range(K):
        test = random_split[j]  # Fold actual como conjunto de prueba
        training_list = random_split[:j] + random_split[j+1:]  # Folds restantes como conjunto de entrenamiento
        training = pd.concat(training_list)

        # Preparar datos de entrenamiento
        X_train = training.drop('_class_', axis=1)
        Y_train = label_encoder.transform(training['_class_'])
        model = LogisticRegression(C=SELECTED_C, class_weight='balanced')
        model.fit(X_train, Y_train)  # Entrenar el modelo

        # Guardar el modelo entrenado en el último pliegue
        if j == K - 1:
            trained_model = model

        # Evaluar en el conjunto de entrenamiento
        X_test = X_train
        Y_test = Y_train
        current_results = utils.estimate_model_performance(model, X_test, Y_test)
        for i in range(len(training_results)):
            training_results[i] += current_results[i]

        # Evaluar en el conjunto de prueba
        X_test = test.drop('_class_', axis=1)
        Y_test = label_encoder.transform(test['_class_'])
        current_results = utils.estimate_model_performance(model, X_test, Y_test)
        for i in range(len(test_results)):
            test_results[i] += current_results[i]

    # Promediar resultados
    for i in range(len(test_results)):
        training_results[i] /= K
        test_results[i] /= K

    # Escribir resultados de entrenamiento
    line = 'Training: precision = {}; recall = {}; f1_score = {}; accuracy = {}'.format(
        training_results[0], training_results[1], training_results[2], training_results[3])
    output.write(line + '\n')
    output_info += f'\n{line}'

    # Escribir resultados de prueba
    line = 'Test: precision = {}; recall = {}; f1_score = {}; accuracy = {}'.format(
        test_results[0], test_results[1], test_results[2], test_results[3])
    output.write(line + '\n')
    output_info += f'\n{line}'

    output.close()
    return output_info

# Evaluar modelos para los datasets binarios
bow = evaluate_binary_model('bow-2', '2-bow-lr')
tfidf = evaluate_binary_model('tfidf-2', '2-tfidf-lr')

# Imprimir resultados resumidos
print(f'{bow}\n\n{tfidf}')


# Guardar el modelo entrenado en un archivo pickle

# import pickle

# with open('lr_model.pkl', 'wb') as file:
#     pickle.dump(trained_model, file)

# print("Modelo guardado en 'lr_model.pkl'")
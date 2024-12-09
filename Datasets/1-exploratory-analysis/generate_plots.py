import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lista de clases de requisitos
clases = ['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']

# Función para obtener el conteo de cada clase en un DataFrame
def obtener_conteos(df, nombre_columna_clase):
    conteos = {clase: 0 for clase in clases}
    for clase in conteos.keys():
        conteos[clase] = df[df[nombre_columna_clase] == clase].shape[0]
    return conteos

# Leer los conjuntos de datos y calcular los conteos
nfr_df = pd.read_csv('./data/nfr.csv', sep=',', header=0, quotechar='"')
conteos_nfr = obtener_conteos(nfr_df, 'class')

promise_exp_df = pd.read_csv('./data/PROMISE_exp.csv', sep=',', header=0, quotechar='"', doublequote=True)
conteos_promise_exp = obtener_conteos(promise_exp_df, '_class_')

# Calcular valores para la gráfica de todas las clases
valores_nfr = [conteos_nfr[clase] for clase in clases]
valores_promise_exp = [conteos_promise_exp[clase] for clase in clases]
diferencias_promise_exp = [a2 - a1 for a1, a2 in zip(valores_nfr, valores_promise_exp)]
indices = np.arange(len(clases))
ancho_barra = 0.45

# Crear gráfica de barras para todas las clases
barras_nfr = plt.bar(indices, valores_nfr, ancho_barra, label='NFR')
barras_diferencias = plt.bar(indices, diferencias_promise_exp, ancho_barra, bottom=valores_nfr, label='PROMISE_exp')

# Configurar la gráfica
plt.title('Distribución de clases en PROMISE_exp')
plt.xlabel('Tipo de requisito')
plt.ylabel('Cantidad')
plt.xticks(indices, clases)
plt.yticks(np.arange(0, 451, 50))
plt.legend()

# Guardar la gráfica
plt.savefig('./grafico_12_clases.png')
plt.close()

# Calcular valores para la gráfica de FR (Funcionales) vs NFR (No funcionales)
conteo_nfr_total = sum([conteos_nfr[clase] for clase in clases[1:]])
valores_nfr = [conteos_nfr['F'], conteo_nfr_total]

conteo_promise_exp_total = sum([conteos_promise_exp[clase] for clase in clases[1:]])
diferencias_promise_exp = [
    conteos_promise_exp['F'] - conteos_nfr['F'],
    conteo_promise_exp_total - conteo_nfr_total
]

# Crear gráfica de barras para FR vs NFR
indices = np.arange(2)
ancho_barra = 0.6

barras_nfr = plt.bar(indices, valores_nfr, ancho_barra, label='NFR')
barras_diferencias = plt.bar(indices, diferencias_promise_exp, ancho_barra, bottom=valores_nfr, label='PROMISE_exp')

# Configurar la gráfica
plt.title('Distribución de clases en PROMISE_exp - FR vs NFR')
plt.xlabel('Tipo de requisito')
plt.ylabel('Cantidad')
plt.xticks(indices, ['FR', 'NFR'])
plt.yticks(np.arange(0, 501, 50))
plt.legend()

# Ajustar tamaño de la figura y guardar la gráfica
figura = plt.gcf()
figura.set_size_inches(4.5, 4.8)
figura.savefig('./grafico_FR_vs_NFR.png')

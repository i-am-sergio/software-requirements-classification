import pandas as pd
from collections import defaultdict
from nltk.corpus import wordnet as wn, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

import nltk

# Descargar los recursos necesarios de NLTK
nltk.download('wordnet')  # Diccionario para lematización
nltk.download('stopwords')  # Lista de palabras irrelevantes
nltk.download('punkt')  # Tokenización de texto
nltk.download('averaged_perceptron_tagger')  # Etiquetado gramatical (POS tagging)

# Leer el archivo CSV con los datos de requisitos
df = pd.read_csv(
    '../Datasets/ForTraining/PROMISE_exp.csv',  # Ruta al archivo CSV
    sep=',',  # Separador de columnas
    header=0,  # Indica que la primera fila contiene los encabezados
    quotechar='"',  # Carácter para manejar comillas
    doublequote=True  # Permitir comillas dobles dentro de campos
)

# Eliminar columnas irrelevantes (en este caso, 'ProjectID')
if 'ProjectID' in df.columns:
    del df['ProjectID']

def process_requirement_text(text):
    """
    Normaliza el texto de un requisito:
    - Convierte a minúsculas.
    - Tokeniza el texto (divide en palabras o símbolos de puntuación).
    - Filtra palabras no alfabéticas y palabras comunes (stopwords).
    - Realiza lematización para obtener la forma base de cada palabra.
    """
    # Convertir a minúsculas y tokenizar
    tokens = word_tokenize(text.lower())
    
    # Inicializar la lematización y las palabras irrelevantes
    lemmatizer = WordNetLemmatizer()
    english_stopwords = stopwords.words('english')
    
    # Mapa de etiquetas gramaticales para lematización
    tag_map = defaultdict(lambda: wn.NOUN)  # Por defecto, tratar como sustantivo
    tag_map['J'] = wn.ADJ  # Adjetivo
    tag_map['V'] = wn.VERB  # Verbo
    tag_map['R'] = wn.ADV  # Adverbio
    
    # Procesar cada token con su etiqueta gramatical
    processed_words = []
    for word, tag in pos_tag(tokens):
        # Filtrar palabras no alfabéticas y stopwords
        if word.isalpha() and word not in english_stopwords:
            # Lematizar palabra según su etiqueta gramatical
            processed_words.append(lemmatizer.lemmatize(word, tag_map[tag[0]]))
    
    # Reconstruir el texto procesado
    return ' '.join(processed_words)

# Aplicar la normalización a cada texto en la columna 'RequirementText'
df['RequirementText'] = df['RequirementText'].apply(process_requirement_text)

# Guardar el DataFrame normalizado en un nuevo archivo CSV
df.to_csv(
    './output/dataset_normalized.csv',  # Ruta y nombre del archivo de salida
    sep=',',  # Separador de columnas
    header=True,  # Incluir encabezados en el archivo
    index=False,  # Excluir índice numérico
    quotechar='"',  # Carácter para manejar comillas
    doublequote=True  # Permitir comillas dobles dentro de campos
)

print("===== Normalizacion completada. Output: './output/dataset_normalized.csv' =====")

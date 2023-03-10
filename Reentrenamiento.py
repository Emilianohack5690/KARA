import tensorflow as tf
import numpy as np
import os
import spacy

# Cargar el modelo de idioma en español
nlp = spacy.load("es_core_news_sm")

# Definir la longitud de la secuencia de entrada y de salida
seq_length = 50

# Descargar los datos de SpaCy si no están ya instalados
spacy.cli.download("es_core_news_sm")

# Directorio que contiene los archivos de texto originales y los nuevos archivos de texto
data_dir = 'entrenamiento'

# Obtener una lista de todos los archivos en el directorio de datos
files = os.listdir(data_dir)

# Concatenar todo el texto en un solo string
corpus = ''
for file in files:
    with open(os.path.join(data_dir, file), 'r', encoding="utf-8") as f:
        text = f.read()
        corpus += text

# Tokenizar el corpus utilizando el tokenizer de SpaCy
tokens = [token.text for token in nlp(corpus)]

# Crear un diccionario de índices para cada token único
token_to_idx = {token: idx for idx, token in enumerate(set(tokens))}

# Convertir la lista de tokens a una lista de índices
idxs = [token_to_idx[token] for token in tokens]

# Crear los pares de entrada/salida
inputs = []
outputs = []
for i in range(len(idxs) - seq_length):
    inputs.append(idxs[i:i+seq_length])
    outputs.append(idxs[i+seq_length])

# Convertir los pares de entrada/salida a arrays de numpy
inputs = np.array(inputs)
outputs = np.array(outputs)

# Cargar el modelo entrenado previamente
modelo=input("Ingresar un modelo entrenado: ")
model = tf.keras.models.load_model(modelo)

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Entrenar el modelo con los nuevos datos
model.fit(inputs, outputs, epochs=30, batch_size=16)

# Guardar el modelo entrenado
save=input("Nombre a guardar nuevo archivo: ")
model.save(save)
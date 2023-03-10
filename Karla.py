import tensorflow as tf
import numpy as np
import os
import string
import spacy

# Descargar el modelo de idioma en español si no está ya instalado
spacy.cli.download("es_core_news_sm")

# Cargar el modelo de idioma en español
nlp = spacy.load("es_core_news_sm")

# Directorio que contiene los archivos de texto
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

# Definir la longitud de la secuencia de entrada y de salida
seq_length = 50

# Crear los pares de entrada/salida
inputs = []
outputs = []
for i in range(len(idxs) - seq_length):
    inputs.append(idxs[i:i+seq_length])
    outputs.append(idxs[i+seq_length])

# Convertir los pares de entrada/salida a arrays de numpy
inputs = np.array(inputs)
outputs = np.array(outputs)

# Definir el modelo de lenguaje natural
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(token_to_idx), 128, input_length=seq_length),
    tf.keras.layers.LSTM(256),
    tf.keras.layers.Dense(len(token_to_idx), activation='softmax')
])

# Compilar el modelo
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["Accuracy"])

# Entrenar el modelo
model.fit(inputs, outputs, epochs=30, batch_size=16, steps_per_epoch=5000)
save=input("Nombre a guardar el modelo: ")
model.save(save)
print("modelo Guardado con Exito")
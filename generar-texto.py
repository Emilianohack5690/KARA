import spacy
import numpy as np
import random

# Cargar el modelo de idioma en español
nlp = spacy.load("es_core_news_sm")

# Definir la longitud de la secuencia de entrada
seq_length = 50

# Cargar el modelo entrenado
modelo=input("ingresa el modelo entrenado: ")
model = tf.keras.models.load_model(modelo)

# Definir la frase semilla
semilla=input("ingresa prompt: ")
seed_text = semilla

# Tokenizar la frase semilla utilizando el tokenizer de SpaCy
seed_tokens = [token.text for token in nlp(seed_text)]

# Convertir la lista de tokens a una lista de índices
seed_idxs = [token_to_idx[token] for token in seed_tokens]

# Generar texto
generated_text = seed_tokens
for i in range(20):
    # Tomar los últimos seq_length tokens como entrada para el modelo
    input_seq = np.array(seed_idxs[-seq_length:])[np.newaxis, :]
    
    # Predecir las probabilidades de los siguientes tokens
    probs = model.predict(input_seq)[0]
    
    # Elegir el token con la probabilidad más alta
    next_idx = np.argmax(probs)
    next_token = list(token_to_idx.keys())[list(token_to_idx.values()).index(next_idx)]
    
    # Agregar el token a la lista de tokens generados
    generated_text.append(next_token)
    
    # Agregar el índice del token a la lista de índices de tokens generados
    seed_idxs.append(next_idx)
    
# Unir la lista de tokens generados en una sola cadena de texto
generated_text = ' '.join(generated_text)

# Imprimir el texto generado
print(generated_text)
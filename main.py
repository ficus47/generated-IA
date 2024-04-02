import os
from PIL import Image
import numpy as np
import gzip

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

def generate_image(model, tokenizer, text, image_width, image_height):
    # Tokenisation du texte
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen)
    
    # Génération de l'image
    prediction = model.predict(padded_sequence)
    prediction = prediction.reshape(image_width, image_height, 3) * 255  # Dénormalisation
    
    # Création de l'image à partir de la prédiction
    generated_image = Image.fromarray(prediction.astype(np.uint8))
    
    return generated_image

    return tf.image.ssim(y_true, y_pred, max_val=255.0)


image_width, image_height = 150, 200

# Chargement des images
image_dir = "train"
images = []
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        img = Image.open(os.path.join(image_dir, filename))
        img = img.convert("RGB")  # Conversion en RGB
        img = img.resize((image_width, image_height))  # Redimensionnement
        img_array = np.array(img)
        images.append(img_array)

for filename in os.listdir("temporaire"):
    if filename.endswith(".jpg"):
        img = Image.open(os.path.join("temporaire", filename))
        img = img.convert("RGB")  # Conversion en RGB
        img = img.resize((image_width, image_height))  # Redimensionnement
        img_array = np.array(img)
        for i in range(10):
           images.append(img_array)

# Lecture du fichier texte
texts = []
with open("text.txt", "r") as file:
  for i in file:
     texts.append(i.split(",")[3])

buf = []

with open("texts.txt", "r") as file:
  for i in file:
     buf.append(i)

buf *= 10
texts += buf

texts = texts[:len(os.listdir(image_dir))]

x = 200
texts, images = texts[:x], images[:x]

# Conversion en array numpy
texts = np.array(texts)
images = np.array(images)

# Normalisation des images
images = images / 255.0

# Padding des textes
maxlen = max(len(seq) for seq in texts)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
texts = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=maxlen)

# Redimensionnement des images pour le LSTM
images = images.reshape(images.shape[0], image_width*image_height*3, 1)


# Vérification des dimensions
print("Textes shape:", texts.shape)
print("Images shape:", images.shape)

# Création du modèle
model = Sequential([
    LSTM(500, input_shape=(maxlen, 1), return_sequences=True),
    Dropout(0.3),
    LSTM(500),

    Dense(image_width*image_height*3, activation="sigmoid")
])

model.compile(optimizer="adam", loss="mse", metrics=["accuracy", "mae", "mse"])
model.summary()

model.fit(texts, images, epochs=200, batch_size=32)

user = ""

model.save("model.keras")

while user != "exit":
  user = input("vous : ")
  image = generate_image(model, tokenizer, user, image_width, image_height)

  image.save("generated_image.jpeg")

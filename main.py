import os
from PIL import Image
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense 
from tensorflow.keras.models import Sequential

# Lecture du fichier texte
texts = []
with open("texts.txt", "r") as file:
    for line in file:
        texts.append(line.strip())

image_width, image_height = 500, 500

# Chargement des images
image_dir = "temporaire"
images = []
for filename in os.listdir(image_dir):
    if filename.endswith(".jpeg"):
        img = Image.open(os.path.join(image_dir, filename))
        img = img.convert("RGB")  # Conversion en RGB
        img = img.resize((image_width, image_height))  # Redimensionnement
        img_array = np.array(img)
        images.append(img_array)

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
images = images.reshape(images.shape[0], image_width*image_height, 3)


# Vérification des dimensions
print("Textes shape:", texts.shape)
print("Images shape:", images.shape)

# Création du modèle
model = Sequential([
    LSTM(500, input_shape=(maxlen, 6)),
    Dense(image_width*image_height*3, activation="sigmoid")
])

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"], batch_size=16)

model.fit(texts, images, epochs=100)

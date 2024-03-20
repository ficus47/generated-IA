import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Définition des paramètres
latent_dim = 100
height = 28
width = 28
channels = 1

# Générateur
generator_input = keras.Input(shape=(latent_dim,))
x = layers.Dense(128 * 7 * 7)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((7, 7, 128))(x)
x = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2DTranspose(128, 4, strides=2, padding="same")(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(1, 7, padding="same", activation="sigmoid")(x)
generator = keras.models.Model(generator_input, x)

# Discriminateur
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(64, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 3, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 3, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation="sigmoid")(x)
discriminator = keras.models.Model(discriminator_input, x)

# Compilation du modèle discriminatoire
discriminator_optimizer = keras.optimizers.RMSprop(
    learning_rate=0.0008,
    clipvalue=1.0,
    decay=1e-8
)
discriminator.compile(optimizer=discriminator_optimizer, loss="binary_crossentropy")

# Générateur GAN
discriminator.trainable = False
gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

# Compilation du modèle GAN
gan_optimizer = keras.optimizers.RMSprop(learning_rate=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss="binary_crossentropy")

# Entraînement du GAN
(x_train, _), (_, _) = keras.datasets.mnist.load_data()
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype("float32") / 255.0

iterations = 10000
batch_size = 20
save_dir = "gan_images"

start = 0
for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    generated_images = generator.predict(random_latent_vectors)

    stop = start + batch_size
    real_images = x_train[start:stop]
    combined_images = np.concatenate([generated_images, real_images])

    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += 0.05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)

    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0

    if step % 100 == 0:
        gan.save_weights("gan.h5")

        print("Step %d - Discriminator Loss: %.4f, Adversarial Loss: %.4f" % (step, d_loss, a_loss))

        img = tf.keras.preprocessing.image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, "generated_image_%d.png" % step))


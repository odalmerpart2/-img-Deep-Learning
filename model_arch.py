"""
model_arch.py
--------------
Este archivo NO estaba en el repositorio original. Se agregó como parte de
las modificaciones pedidas en la actividad.

¿Por qué existe este archivo?
------------------------------
El repositorio original guarda el modelo entrenado usando
`model.save('trainedmodels/vgg16_....h5')`, con una versión antigua de
Keras/TensorFlow. Al intentar abrir ese archivo directamente con
`load_model(...)` usando una versión moderna de TensorFlow/Keras (la que
se instala hoy en día con `pip install tensorflow`), ocurre un error de
incompatibilidad:

    ValueError: Kernel shape must have the same length as input, but
    received kernel of shape (3, 3, 3, 1) and input of shape
    (None, None, 224, 224, 3).

Esto pasa porque el formato de guardado "legacy" .h5 no reconstruye bien
la primera capa (Conv2D con input_shape) en las versiones nuevas de Keras.

La solución que se implementó fue:
1. Reconstruir aquí, en código, la MISMA arquitectura que usa
   `CatDogTraining.py` en este repositorio (capa por capa, exactamente
   igual).
2. Cargar únicamente los PESOS guardados
   (`vgg16_epoch_13_accuracy_84.55_weights_.h5`) sobre esa arquitectura,
   en lugar de cargar el modelo completo. Esto sí funciona porque los
   pesos se guardan por separado y no dependen del formato de
   configuración del modelo que causaba el error.

Así, tanto `predict.py` como cualquier script de prueba pueden importar
`build_model()` y `load_pretrained_model()` desde este archivo para
evitar repetir la arquitectura en cada script.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, GaussianNoise, MaxPooling2D, Flatten, Dense, Dropout
)

# Tamaño de imagen que espera el modelo (el mismo que usa CatDogTraining.py)
IMG_SIZE = 224

# Nombres de las clases en el mismo orden en que Keras las asignó durante
# el entrenamiento original (alfabético: cat=0, dog=1). Esto se confirma
# imprimiendo train_generator.class_indices en CatDogTraining.py.
CLASS_NAMES = ["Gato", "Perro"]

# Rutas por defecto a los archivos del modelo pre-entrenado incluido en el
# repositorio original.
DEFAULT_WEIGHTS_PATH = "trainedmodels/vgg16_epoch_13_accuracy_84.55_weights_.h5"


def build_model():
    """
    Reconstruye, capa por capa, la misma arquitectura de red neuronal
    convolucional definida en CatDogTraining.py de este repositorio.
    Devuelve el modelo SIN pesos entrenados (recién inicializado).
    """
    model = Sequential()
    model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 3)))

    model.add(Conv2D(1, kernel_size=3, padding="same"))
    model.add(GaussianNoise(0.25))

    model.add(Conv2D(8, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(16, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(64, kernel_size=3, padding="same", activation="relu"))
    model.add(GaussianNoise(0.25))
    model.add(Conv2D(128, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(256, kernel_size=3, padding="same", activation="relu"))
    model.add(GaussianNoise(0.25))
    model.add(Conv2D(512, kernel_size=3, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.1))
    model.add(GaussianNoise(0.25))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(len(CLASS_NAMES), activation="sigmoid"))

    return model


def load_pretrained_model(weights_path=DEFAULT_WEIGHTS_PATH):
    """
    Construye la arquitectura y le carga los pesos entrenados del
    repositorio original. Devuelve el modelo listo para predecir.
    """
    model = build_model()
    model.load_weights(weights_path)
    return model

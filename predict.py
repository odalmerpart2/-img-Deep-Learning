"""
predict.py
----------
Archivo NUEVO agregado al repositorio original "DeepLearning-Classifier"
como parte del desarrollo requerido de la actividad "Clasificación de
Imágenes con Deep Learning".

El repositorio original (CatDogTraining-2) solo permitía ENTRENAR un
modelo (CatDogTraining.py) y probarlo contra una carpeta fija de imágenes
(CatDogTest.py). No existía ninguna forma de indicarle al programa la
ruta de UNA imagen nueva desde la terminal.

Este script agrega justamente eso: recibe la ruta de una imagen (JPG o
PNG) por línea de comandos, la preprocesa, la pasa por el modelo ya
entrenado y muestra en pantalla y en una ventana gráfica el resultado.

Modo de uso desde la terminal:
    python predict.py --image ruta/de/la/imagen.jpg

Ejemplo:
    python predict.py --image dataset/testImage/cat1.jpg
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model_arch import (
    IMG_SIZE,
    CLASS_NAMES,
    DEFAULT_WEIGHTS_PATH,
    load_pretrained_model,
)


def parsear_argumentos():
    """Define y lee los argumentos que se pasan por la terminal."""
    parser = argparse.ArgumentParser(
        description="Clasifica una imagen como Gato o Perro usando el "
                    "modelo entrenado en este repositorio."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Ruta de la imagen a clasificar (formato .jpg o .png).",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS_PATH,
        help="Ruta a los pesos del modelo entrenado "
             f"(por defecto: {DEFAULT_WEIGHTS_PATH}).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="No abrir una ventana gráfica con el resultado (útil si se "
             "ejecuta en un servidor sin pantalla). El resultado en texto "
             "siempre se imprime en la terminal.",
    )
    return parser.parse_args()


def cargar_y_preprocesar_imagen(ruta_imagen, tamano=IMG_SIZE):
    """
    Carga una imagen JPG/PNG desde disco y la deja lista para el modelo:

    1. Abre la imagen y la convierte a RGB (por si viene en escala de
       grises o con canal de transparencia).
    2. La redimensiona al tamaño que espera el modelo (224x224, el mismo
       tamaño usado en CatDogTraining.py).
    3. Convierte los píxeles a float32 y los dejar en formato de arreglo
       de numpy con forma (1, 224, 224, 3), lista para model.predict().

    NOTA IMPORTANTE sobre la normalización:
    En CatDogTraining.py del repositorio original, el generador de
    imágenes de entrenamiento se crea así:

        train_datagen = ImageDataGenerator()

    Es decir, SIN el parámetro `rescale=1./255`. Esto significa que el
    modelo fue entrenado con los píxeles en su rango original [0, 255],
    no normalizados a [0, 1]. Se comprobó en la práctica que si aquí se
    normaliza dividiendo entre 255 (lo más común en otros proyectos), las
    predicciones del modelo pre-entrenado se vuelven incorrectas, porque
    el modelo nunca vio píxeles en ese rango durante el entrenamiento.

    Por eso, para que las predicciones sean coherentes con el modelo que
    trae este repositorio, aquí la "normalización" consiste en convertir
    los píxeles a float32 en su escala original [0, 255], que es
    precisamente la escala de valores con la que se entrenó la red.
    """
    imagen = Image.open(ruta_imagen).convert("RGB")
    imagen_redimensionada = imagen.resize((tamano, tamano))

    arreglo = np.array(imagen_redimensionada).astype("float32")
    lote = np.expand_dims(arreglo, axis=0)  # forma (1, 224, 224, 3)

    return imagen_redimensionada, lote


def predecir(modelo, lote_imagen):
    """
    Ejecuta la predicción sobre la imagen ya preprocesada y devuelve:
    - el nombre de la clase predicha ("Gato" o "Perro")
    - la probabilidad/confianza de esa predicción (como porcentaje)
    """
    salida = modelo.predict(lote_imagen, verbose=0)[0]  # ej: [0.94, 0.03]

    indice_predicho = int(np.argmax(salida))
    clase_predicha = CLASS_NAMES[indice_predicho]

    # La capa de salida usa activación 'sigmoid' con una neurona por
    # clase (en vez de 'softmax'), tal como está definido en
    # CatDogTraining.py de este repositorio. Esto significa que los dos
    # valores de salida no necesariamente suman 1. Para reportar una
    # confianza que sí se pueda leer como porcentaje (0-100%), se
    # normaliza dividiendo el valor de la clase ganadora entre la suma de
    # ambos valores.
    confianza = float(salida[indice_predicho] / np.sum(salida)) * 100

    return clase_predicha, confianza, salida


def mostrar_resultado(imagen_mostrar, clase_predicha, confianza, ruta_imagen, mostrar_ventana=True):
    """
    Imprime el resultado en la terminal y muestra la imagen junto con la
    predicción en una ventana gráfica (usando matplotlib).
    """
    nombre_archivo = os.path.basename(ruta_imagen)

    print("\n----- Resultado de la predicción -----")
    print(f"Imagen analizada : {nombre_archivo}")
    print(f"Clase predicha   : {clase_predicha}")
    print(f"Confianza        : {confianza:.2f}%")
    print("---------------------------------------\n")

    plt.figure(figsize=(5, 5))
    plt.imshow(imagen_mostrar)
    plt.axis("off")
    plt.title(f"Predicción: {clase_predicha}  ({confianza:.2f}% de confianza)")
    plt.tight_layout()

    # Se guarda siempre una copia del resultado como imagen PNG, útil
    # para incluir en el documento de pruebas que pide la actividad.
    nombre_salida = f"resultado_{os.path.splitext(nombre_archivo)[0]}.png"
    plt.savefig(nombre_salida, dpi=150)
    print(f"Se guardó una imagen con el resultado en: {nombre_salida}")

    if mostrar_ventana:
        plt.show()
    else:
        plt.close()


def main():
    args = parsear_argumentos()

    if not os.path.isfile(args.image):
        print(f"Error: no se encontró el archivo de imagen '{args.image}'")
        sys.exit(1)

    extension = os.path.splitext(args.image)[1].lower()
    if extension not in (".jpg", ".jpeg", ".png"):
        print(f"Error: formato de imagen no soportado '{extension}'. "
              "Usa un archivo .jpg o .png")
        sys.exit(1)

    print("Cargando el modelo entrenado...")
    modelo = load_pretrained_model(args.weights)

    print(f"Procesando la imagen: {args.image}")
    imagen_mostrar, lote_imagen = cargar_y_preprocesar_imagen(args.image)

    clase_predicha, confianza, salida_completa = predecir(modelo, lote_imagen)

    mostrar_resultado(
        imagen_mostrar,
        clase_predicha,
        confianza,
        args.image,
        mostrar_ventana=not args.no_show,
    )


if __name__ == "__main__":
    main()

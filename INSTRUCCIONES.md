# Clasificación de Imágenes con Deep Learning — Instrucciones

Este proyecto es una versión modificada de `CatDogTraining-2`, dentro del
repositorio original
[DeepLearning-Classifier](https://github.com/eeyribas/DeepLearning-Classifier).

## 1. Clonar y entrar a la carpeta

```bash
git clone https://github.com/eeyribas/DeepLearning-Classifier.git
cd DeepLearning-Classifier/CatDogTraining-2
```

Luego reemplaza/agrega en esa carpeta los archivos incluidos en esta
entrega: `model_arch.py`, `predict.py`, `requirements.txt`, y el
`CatDogTest.py` modificado.

## 2. Crear y activar un entorno virtual

```bash
python3 -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## 4. Ejecutar el proyecto original (evidencia)

```bash
python CatDogTest.py
```

Esto carga el modelo ya entrenado que trae el repositorio
(`trainedmodels/vgg16_epoch_13_accuracy_84.55.h5`, 84.55% de accuracy
original) y lo prueba contra las imágenes en `dataset/testImage`.

> **Nota sobre un problema real encontrado:** el archivo `.h5` original no
> carga con `load_model()` en versiones actuales de TensorFlow/Keras
> (error `ValueError: Kernel shape must have the same length as input...`)
> porque se guardó con una versión antigua de Keras. La solución
> implementada (ver `model_arch.py`) fue reconstruir la arquitectura de
> red en código y cargar solo los pesos con `load_weights()`, que sí son
> compatibles. Este cambio ya está aplicado en `CatDogTest.py` y en
> `predict.py`.

## 5. Ejecutar predict.py (archivo obligatorio, nuevo)

```bash
python predict.py --image dataset/testImage/cat1.jpg
```

Esto imprime en la terminal la clase predicha (Gato/Perro) y el
porcentaje de confianza, y abre una ventana con la imagen y el
resultado. También guarda automáticamente una copia como
`resultado_<nombre_de_archivo>.png`.

Para usar tu propia imagen:

```bash
python predict.py --image ruta/a/tu/imagen.jpg
```

Si estás en un servidor sin pantalla (sin entorno gráfico), agrega
`--no-show` para que no intente abrir una ventana:

```bash
python predict.py --image ruta/a/tu/imagen.jpg --no-show
```

## Archivos nuevos o modificados en esta entrega

| Archivo | Estado | Descripción |
|---|---|---|
| `predict.py` | Nuevo (obligatorio) | Clasifica una imagen nueva desde la terminal |
| `model_arch.py` | Nuevo | Arquitectura de red + carga de pesos (soluciona incompatibilidad) |
| `requirements.txt` | Nuevo | Dependencias del proyecto |
| `CatDogTest.py` | Modificado | Usa `model_arch.py` para cargar el modelo correctamente |
| `INSTRUCCIONES.md` | Nuevo | Este archivo |

import os
import sys
import cv2
from datetime import datetime
import time
import numpy as np

# --- MODIFICACIÓN ---
# La línea original de este archivo era:
#     model = load_model('trainedmodels/vgg16_epoch_13_accuracy_84.55.h5')
# Esa línea falla con las versiones actuales de TensorFlow/Keras porque el
# archivo .h5 fue guardado con una versión antigua de Keras y el formato
# "legacy" de guardado ya no reconstruye bien la primera capa Conv2D. El
# error que produce es:
#   ValueError: Kernel shape must have the same length as input...
#
# La solución (ver model_arch.py) es reconstruir la misma arquitectura de
# red en código y cargar solamente los pesos, que sí son compatibles.
from model_arch import load_pretrained_model

try:
    class_indices = ['cat', 'dog']
    model = load_pretrained_model()
    folder = 'dataset/testImage'
    images_count = len(os.listdir(folder))
    print('file count : ', images_count)

    for file in os.listdir(folder):
        path = folder + '/' + file
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (224, 224))
        img = np.array([img]).reshape((1, 224, 224, 3))
        res = model.predict(img)
        pred_name = class_indices[np.argmax(res)]
        print('result: ', pred_name, 'file: ', file)

except Exception as e:
    print(str(e))
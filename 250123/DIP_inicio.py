import numpy as np
from matplotlib import pyplot as plt
import tensorflow
import sklearn
import jupyter
import cv2

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Mostrar imagen
image = cv2.imread(r'C:\Users\karen\OneDrive\Escritorio\DIP\250123\brain.png')
cv2.imshow('Imagen', image)
cv2.waitKey(0)

# Tamaño
print(image.shape)
h, w, c = image.shape
print('Width:  ', w)
print('Height: ', h)
print('Channels:', c)

# Histograma para cada canal
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.figure(figsize=(8, 6))
plt.bar(range(256), hist[:, 0], width=1, color='blue', alpha=0.7)
plt.title('Histograma')
plt.xlabel('Valores de intensidad')
plt.ylabel('Número de píxeles')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Brillo (promedio de intensidad)
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
brillo = np.mean(grayscale)
print('Brillo promedio:', brillo)

# Resolución
resolucion = h * w
print('Resolución (pixeles totales):', resolucion)

# Determinar los niveles de cuantización
niveles_unicos = np.unique(image)  # Encontrar valores únicos
print("Niveles de cuantización:", len(niveles_unicos))
print("Valores de cuantización únicos:", niveles_unicos)

cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas por OpenCV

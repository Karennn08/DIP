import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import sklearn
import jupyter
import cv2

# Mostrar imagen
image = cv2.imread(r'C:\Users\karen\OneDrive\Escritorio\DIP\250123\brain.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Imagen', image)

# Tamaño
print(image.shape)

# Resolución
h, w = image.shape
print('Resolucion:', w, 'x', h)

# Histograma para cada canal
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.bar(range(256), hist[:, 0], width=1, color='gray', alpha=0.7)
plt.title('Histograma')
plt.xlabel('Valores de intensidad')
plt.ylabel('Número de píxeles')
plt.show()

# Brillo (promedio de intensidad)
bright = np.mean(image)
print('Brillo promedio:', bright)

# Cuantización
unique_levels = np.unique(image) # Obtener los valores únicos de la imagen
num_levels = len(unique_levels) # Contar los niveles únicos
print('Cuantizacion:', num_levels)

cv2.waitKey(0)
cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas por OpenCV

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import sklearn
import jupyter
import cv2

# Show image
image = cv2.imread(r'C:\Users\karen\OneDrive\Escritorio\DIP\250123\brain.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Imagen', image)

# Size
print(image.shape)

# Resolution
h, w = image.shape
print('Resolucion:', w, 'x', h)

# Histogram
hist = cv2.calcHist([image], [0], None, [256], [0, 256])
plt.bar(range(256), hist[:, 0], width=1, color='gray', alpha=0.7)
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Pixels')
plt.show()

# Bright 
bright = np.mean(image)
print('Bright:', bright)

# Quantization
unique_levels = np.unique(image) # Obtener los valores únicos de la imagen
num_levels = len(unique_levels) # Contar los niveles únicos
print('Quantization:', num_levels)

# Contrast 
contrast = image.std()
print('Contrast:', contrast)

cv2.waitKey(0)
cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas por OpenCV

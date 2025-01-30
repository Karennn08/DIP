import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import sklearn
import jupyter
import cv2


# Show image
us_image = cv2.imread(r'C:\Users\karen\OneDrive\Escritorio\DIP\UltraSound\a22.jpg')
us_label = cv2.imread(r'C:\Users\karen\OneDrive\Escritorio\DIP\UltraSound\a22.png')

cv2.imshow('Imagen', us_image)
cv2.waitKey(0)
cv2.imshow('Imagen', us_label)
cv2.waitKey(0)

# Resolution
h, w, c = us_image.shape
print('Resolucion:', h, 'x', w , ' canales ', c)

# Histogram
hist = cv2.calcHist([us_image], [0], None, [256], [0, 256])
plt.bar(range(256), hist[:, 0], width=1, color='red', alpha=0.7)
plt.title('Histogram')
plt.xlabel('Intensity')
plt.ylabel('Pixels')
plt.show()

# Bright 
bright = np.mean(us_image)
print('Bright:', bright)

# Bit depth
print("Tipo de dato de la imagen:", us_image.dtype)


cv2.waitKey(0)
cv2.destroyAllWindows()  # Cierra todas las ventanas abiertas por OpenCV

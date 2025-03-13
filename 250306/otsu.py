import cv2
import numpy as np

#Otsu algoritm
def binarize_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Aplicar binarización con Otsu
    threshold_value, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Umbral calculado por Otsu:", threshold_value)

    cv2.imshow('Imagen Original', image)
    cv2.imshow('Imagen Binarizada', binary_image)
    
    cv2.waitKey(0)

# Uso
image_path = r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\1-037.JPG"
binarize_image(image_path)

#Region growing
#se elige pixeles semilla y se van comparando con su al rededor a ver si son similares (nivel de gris, color o distancia)
# La 4-vecindad o 4-adyacencia de un píxel p son los 4 píxeles cuyas regiones comparten un lado con p. La 8-vecindad u 8-adyacencia de un píxel p, consiste en los 8 píxeles cuyas regiones comparten un lado o un vértice con p.
# si el pixel es menor al humbral de similitud se convierte en una nueva semilla

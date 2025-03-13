import cv2
import numpy as np

def binarize_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Aplicar binarización con Otsu
    threshold_value, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print("Umbral calculado por Otsu:", threshold_value)

    cv2.imshow('Imagen Original', image)
    cv2.imshow('Imagen Binarizada', binary_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Uso
image_path = r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\1-037.JPG"
binarize_image(image_path)

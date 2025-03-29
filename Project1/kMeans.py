import cv2
import numpy as np

# Cargar imagen en escala de grises  
image_path = r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\1-037.JPG"
#image_path = r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\brain.png"
imagen = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Convertir la imagen en una lista de píxeles (formato necesario para K-Means)
pixeles = imagen.reshape((-1, 1))  # Convertir a un array de una sola columna
pixeles = np.float32(pixeles)  # Convertir a flotante para K-Means

# Definir criterios de K-Means (máximo de 10 iteraciones o cambio menor a 1.0)
criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Número de clusters (ajustable según la imagen)
K = 3  # Tres segmentos (puedes probar con 2, 4, etc.)

# Aplicar K-Means
_, etiquetas, centros = cv2.kmeans(pixeles, K, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convertir los centros a enteros (niveles de gris)
centros = np.uint8(centros)

# Asignar a cada píxel su nivel de gris del cluster correspondiente
imagen_segmentada = centros[etiquetas.flatten()]

# Reconstruir la imagen con las etiquetas de segmentación
imagen_segmentada = imagen_segmentada.reshape(imagen.shape)

# Mostrar imágenes
cv2.imshow("Imagen Original", imagen)
cv2.imshow("Segmentación K-Means", imagen_segmentada)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

def multi_region_growing(img, seed_intensity, threshold=15):
    """
    Segmentación por crecimiento de regiones con múltiples semillas.
    
    - img: imagen en escala de grises.
    - seed_intensity: valor de intensidad inicial para semillas (ejemplo: 100).
    - threshold: diferencia de intensidad permitida en la expansión.
    
    Retorna una máscara binaria con las regiones segmentadas.
    """
    h, w = img.shape
    segmented = np.zeros((h, w), np.uint8)  # Máscara de salida

    # Encontrar todas las semillas (píxeles con la intensidad inicial)
    seeds = np.column_stack(np.where(img == seed_intensity))

    # Usaremos una pila para explorar píxeles (Región Creciente)
    stack = list(map(tuple, seeds))  # Convertir a lista de tuplas (x, y)

    while stack:
        x, y = stack.pop()
        if segmented[x, y] == 0:  # Si aún no ha sido segmentado
            diff = abs(int(img[x, y]) - int(seed_intensity))
            if diff <= threshold:
                segmented[x, y] = 255  # Marcar como segmento
                
                # Añadir vecinos conectados en 4 direcciones (o 8 si se desea)
                if x > 0: stack.append((x - 1, y))
                if x < h - 1: stack.append((x + 1, y))
                if y > 0: stack.append((x, y - 1))
                if y < w - 1: stack.append((x, y + 1))

    return segmented

# Cargar imagen en escala de grises
image_path = r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\1-037.JPG"
imagen = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Seleccionar el valor de intensidad inicial (ejemplo: 100)
seed_intensity = 150  # Ajustar según la imagen
threshold = 50  # Ajustar umbral de tolerancia

# Aplicar la segmentación
segmentacion = multi_region_growing(imagen, seed_intensity, threshold)

# Mostrar resultado
cv2.imshow("Segmentación Multiregión", segmentacion)
cv2.waitKey(0)
cv2.destroyAllWindows()

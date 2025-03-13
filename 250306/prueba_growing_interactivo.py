import cv2
import numpy as np

# Cargar imagen en escala de grises
image_path = r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\1-037.JPG"
imagen = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def multi_region_growing(img, seed_intensity, threshold=15):
    """
    Segmentación por crecimiento de regiones con múltiples semillas.
    - img: imagen en escala de grises.
    - seed_intensity: valor de intensidad inicial (seleccionado por clic).
    - threshold: diferencia de intensidad permitida en la expansión.
    """
    h, w = img.shape
    segmented = np.zeros((h, w), np.uint8)  # Máscara de salida

    # Encontrar todas las semillas (píxeles con la intensidad seleccionada)
    seeds = np.column_stack(np.where(img == seed_intensity))

    # Usaremos una pila para explorar píxeles vecinos
    stack = list(map(tuple, seeds))  # Convertir a lista de tuplas (x, y)

    while stack:
        x, y = stack.pop()
        if segmented[x, y] == 0:  # Si no ha sido segmentado
            diff = abs(int(img[x, y]) - int(seed_intensity))
            if diff <= threshold:
                segmented[x, y] = 255  # Marcar como segmento
                
                # Añadir vecinos en 4 direcciones
                if x > 0: stack.append((x - 1, y))
                if x < h - 1: stack.append((x + 1, y))
                if y > 0: stack.append((x, y - 1))
                if y < w - 1: stack.append((x, y + 1))

    return segmented

def click_event(event, x, y, flags, param):
    """
    Evento de clic para seleccionar una semilla y ejecutar la segmentación.
    """
    if event == cv2.EVENT_LBUTTONDOWN:  # Si se hace clic izquierdo
        seed_intensity = imagen[y, x]  # Obtener intensidad del píxel clickeado
        print(f"Semilla seleccionada en ({x}, {y}) con intensidad: {seed_intensity}")

        # Aplicar segmentación con la intensidad seleccionada
        segmentacion = multi_region_growing(imagen, seed_intensity, threshold=15)
        
        # Mostrar la segmentación
        cv2.imshow("Segmentación con Region Growing", segmentacion)

# Mostrar imagen y activar el evento de clic
cv2.imshow("Selecciona una semilla", imagen)
cv2.setMouseCallback("Selecciona una semilla", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

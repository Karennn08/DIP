import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

# Leer la imagen
a = Image.open(r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\1-037.JPG")
print("Tamaño de la imagen original:", a.size)
plt.imshow(a)
plt.title("Imagen Original")
plt.show()

# Redimensionar la imagen al 30% de su tamaño original
a2 = a.resize((int(a.size[0] * 0.3), int(a.size[1] * 0.3)))
print("Tamaño de la imagen redimensionada:", a2.size)
plt.imshow(a2)
plt.title("Imagen Redimensionada al 30%")
plt.show()

# Redimensionar la imagen utilizando el método "nearest"
ann = a2.resize(a.size, Image.NEAREST)
print("Tamaño de la imagen redimensionada con 'nearest':", ann.size)
plt.imshow(ann)
plt.title("Imagen Redimensionada con 'nearest'")
plt.show()

# Redimensionar la imagen utilizando el método "bilinear"
ab = a2.resize(a.size, Image.BILINEAR)
print("Tamaño de la imagen redimensionada con 'bilinear':", ab.size)
plt.imshow(ab)
plt.title("Imagen Redimensionada con 'bilinear'")
plt.show()

# Redimensionar la imagen utilizando el método "bicubic"
abb = a2.resize(a.size, Image.BICUBIC)
print("Tamaño de la imagen redimensionada con 'bicubic':", abb.size)
plt.imshow(abb)
plt.title("Imagen Redimensionada con 'bicubic'")
plt.show()

# Invertir los colores de la imagen original
L = 255
aneg = Image.eval(a, lambda x: L - x)
print("Tamaño de la imagen con colores invertidos:", aneg.size)
plt.imshow(aneg)
plt.title("Imagen con Colores Invertidos")
plt.show()


#logaritmica
a = np.array(a)
c = 50
alog = c * np.log1p(a.astype(float))  # log1p calcula log(1 + x)
plt.imshow(alog.astype(np.uint8), cmap='gray')
plt.title("Imagen con logaritmo")
plt.show()

#Exponential
r = 0.4;
c = 28;
ag = c*a.astype(float)**r;
plt.imshow(ag.astype(np.uint8), cmap='gray')
plt.title("Imagen con gama")
plt.show()

# Equalizar el histograma
a2_cv = np.array(a2)
gray_image = cv2.cvtColor(a2_cv, cv2.COLOR_RGB2GRAY)
ah = cv2.equalizeHist(gray_image)
plt.imshow(ah, cmap='gray')
plt.title("Imagen Ecualizada")
plt.show()


#IMAGE FILTERING

#son para resaltar caracteristicas de la imagen manipulando sus pixeles, puedo realzar componentes de alta freq y baja freq

#ALTA FREQ = variacion entre pixeles, bordes, detalles finos, texturas
#BAJA FREQ = cambios lentos entre nivele de colores o texturas, fondos, areas homogeneas

#Filtrado espacial esta definido por convolucion: se define a partir de un filtro o kerne o mascara, de un tamaño nxn usualmente n es impar para que haya un pixel central.
#se calcula la suma de los productos entre cada coef del filtro y los pixeles de la imagen original, desplazando la mascara.
#Doble sumatoria, dependiendo de los coeficientes se define pasa alta o bajas

#PASA BAJAS: todos coef positivos y sumar 1 , si es 3x3 hay 9 pixeles, si todos son 1 suman 9 y 9/9 es 1. Entre mas grande sea el filtro mas borroso se ve.

#Hacer filtro promedio, kernet con opencv

#PASA ALTOS: mascaras de derivadas direccionales, sobel, prewitt, 

#PADDING para evitar el ruido de los bordes que resulta al pasar el filtro por esos bordes. Es como agregar pixeles a las imagen pero por fuera, deja el centro igual. HAY DISTINTAS FORMAS DE HACER PADDING...



#FILTRADO EN FRECUENCIA:

#para filtrar en la fecuencia, se hace transf de fourier y luego el producto con los coef del filtro (mascara)
# y luego la inversa para recuperar la imagen

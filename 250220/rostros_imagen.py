import cv2

# Cargar el clasificador Haar pre-entrenado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Leer la imagen
image_path = 'rostros2.jpg'  # Reemplazar con la ruta de tu imagen
img = cv2.imread(image_path)

# Convertir la imagen a escala de grises (requerido por Haar cascades)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detectar rostros
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Dibujar rectángulos alrededor de los rostros detectados
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Mostrar la imagen con los rostros detectados
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

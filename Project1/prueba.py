from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
import cv2
from plyer import filechooser
import numpy as np
from matplotlib import pyplot as plt

class miVentana(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'

        # Agregar imagen de fondo
        with self.canvas.before:
            Color(0, 0, 1, 0.5)
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self.actualizar_fondo, pos=self.actualizar_fondo)

        # Sección izquierda
        self.izquierda = BoxLayout(orientation='vertical', padding=20, spacing=10)

        # Espacios para imágenes
        self.imagen_grande = Image(size_hint=(1, 0.5))

        # Botones subir y guardar
        self.botones_set = GridLayout(cols=2, size_hint=(1, 0.1))

        boton_pequeno1 = Button(text='Cargar Imagen', font_size ="20sp", background_color=(0, 0, 1, 1), size_hint=(0.5, 0.1))
        boton_pequeno1.bind(on_press=self.cargar_imagen)
        self.botones_set.add_widget(boton_pequeno1)

        boton_pequeno2 = Button(text='Guardar Imagen',font_size ="20sp", background_color=(0, 0, 1, 1), size_hint=(0.5, 0.1))
        boton_pequeno2.bind(on_press=self.guardar_imagen)
        self.botones_set.add_widget(boton_pequeno2)

        # Botones de acción
        self.botones_actions = GridLayout(cols=2, size_hint=(1, 0.3))

        operaciones = [
            ("Multiplicación", self.realizar_ThresholdOtsu),
            ("Negativo", self.realizar_Kmeans),
            ("Filtro Gaussiano", self.realizar_RegionGrow),
            ("Pasa alto", self.realizar_Watershed)
        ]

        self.botones = []
        for texto, funcion in operaciones:
            btn = Button(text=texto, font_size ="20sp", background_color=(0, 0, 1, 1))
            btn.bind(on_press=funcion)
            self.botones.append(btn)
            self.botones_actions.add_widget(btn)

        # Agregar widgets
        self.izquierda.add_widget(self.botones_set)
        self.izquierda.add_widget(self.imagen_grande)
        self.izquierda.add_widget(self.botones_actions)

        # Sección derecha
        self.derecha = BoxLayout(orientation='vertical', padding=20, spacing=10)

        self.imagen1 = Image(size_hint=(1, 0.5))
        self.imagen2 = Image(size_hint=(1, 0.5))

        self.derecha.add_widget(self.imagen1)
        self.derecha.add_widget(self.imagen2)

        # Agregar secciones a la ventana principal
        self.add_widget(self.izquierda)
        self.add_widget(self.derecha)

    def actualizar_fondo(self, *args):
        self.rect.size = self.size
        self.rect.pos = self.pos

    def cargar_imagen(self, instance):
        filechooser.open_file(on_selection=self.selection)

    def selection(self, selection):
        self.imagenCV = cv2.imread(selection[0])
        buffer = cv2.flip(self.imagenCV, 0).tobytes()
        textura = Texture.create(size=(self.imagenCV.shape[1], self.imagenCV.shape[0]), colorfmt='bgr')
        textura.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.imagen_grande.texture = textura

    def actualizar_imagen1(self, image):
        buffer = cv2.flip(image, 0).tobytes()
        textura = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        textura.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.imagen1.texture = textura
        self.calcular_histograma(image)

    def calcular_histograma(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        plt.figure()
        plt.bar(range(256), hist[:, 0], width=1, color='gray', alpha=0.7)
        plt.xlim([0, 256])
        plt.savefig("histograma.png")
        plt.close()
        hist_image = cv2.imread("histograma.png")
        buffer = cv2.flip(hist_image, 0).tobytes()
        textura = Texture.create(size=(hist_image.shape[1], hist_image.shape[0]), colorfmt='bgr')
        textura.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.imagen2.texture = textura

    def guardar_imagen(self, instance):
        filechooser.save_file(on_selection=self.save_selection)

    def save_selection(self, selection):
        if selection and self.imagen1.texture:
            width, height = self.imagen1.texture.size
            buffer = self.imagen1.texture.pixels
            image_array = np.frombuffer(buffer, np.uint8).reshape((height, width, 4))
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
            image_array = cv2.flip(image_array, 0)
            cv2.imwrite(selection[0], image_array)



    def realizar_ThresholdOtsu(self, instance):
        pass

    def realizar_Kmeans(self, instance):
        pass

    def realizar_RegionGrow(self, instance):

        # Convertir imagen a escala de grises
        imagen_gris = cv2.cvtColor(self.imagenCV, cv2.COLOR_BGR2GRAY)

        # Variables para almacenar la semilla
        seed_selected = False
        seed_intensity = None
        seed_pos = (0, 0)

        # Función de manejo de clic
        def click_event(event, x, y, flags, param):
            nonlocal seed_selected, seed_intensity, seed_pos
            if event == cv2.EVENT_LBUTTONDOWN:
                seed_intensity = imagen_gris[y, x]
                seed_pos = (x, y)
                print(f"Semilla seleccionada en ({x}, {y}) con intensidad: {seed_intensity}")
                seed_selected = True

        # Mostrar imagen y capturar clic
        cv2.imshow("Selecciona una semilla", imagen_gris)
        cv2.setMouseCallback("Selecciona una semilla", click_event)

        # Esperar hasta que se seleccione la semilla
        while not seed_selected:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return

        # Parámetros de segmentación
        threshold = 15
        h, w = imagen_gris.shape
        segmented = np.zeros((h, w), np.uint8)  # Máscara de salida

        # Encontrar todas las semillas (píxeles con la intensidad seleccionada)
        seeds = np.column_stack(np.where(imagen_gris == seed_intensity))

        # Usar una pila para explorar píxeles vecinos
        stack = list(map(tuple, seeds))  # Convertir a lista de tuplas (x, y)

        while stack:
            x, y = stack.pop()
            if segmented[x, y] == 0:  # Si no ha sido segmentado
                diff = abs(int(imagen_gris[x, y]) - int(seed_intensity))
                if diff <= threshold:
                    segmented[x, y] = 255  # Marcar como segmento

                    # Añadir vecinos en 4 direcciones
                    if x > 0: stack.append((x - 1, y))
                    if x < h - 1: stack.append((x + 1, y))
                    if y > 0: stack.append((x, y - 1))
                    if y < w - 1: stack.append((x, y + 1))

        # Actualizar imagen en el widget de Kivy
        self.actualizar_imagen1(segmented)

        cv2.destroyAllWindows()



    def realizar_Watershed(self, instance):
        pass

class myApp(App):
    def build(self):
        return miVentana()

if __name__ == '__main__':
    myApp().run()

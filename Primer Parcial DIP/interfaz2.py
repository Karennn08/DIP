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

        # Sección izquierda
        self.izquierda = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Espacios para imágenes
        self.imagen_grande = Image(size_hint=(1, 0.5))
        self.boton_pequeno1 = Button(text=f'Cargar Imagen', background_color=(1, 0, 0, 1), size_hint=(0.5, 0.1))
        self.boton_pequeno1.bind(on_press=self.cargar_imagen)
        
        # Botones de acción
        self.botones_layout = GridLayout(cols=2, size_hint=(1, 0.3))

        # Lista operaciones
        operaciones = [
            ("Multiplicación", self.realizar_multiplicacion),
            ("Negativo", self.realizar_negativo),
            ("Filtro Gaussiano", self.realizar_gauss),
            ("Pasa alto", self.realizar_pasaAlto)]

        # Generar botones
        self.botones = []
        for texto, funcion in operaciones:
            btn = Button(text=texto, background_color=(1, 0, 0, 1))
            btn.bind(on_press=funcion)
            self.botones.append(btn)
            self.botones_layout.add_widget(btn)

        # Agregar widgets
        self.izquierda.add_widget(self.boton_pequeno1)
        self.izquierda.add_widget(self.imagen_grande)
        self.izquierda.add_widget(self.botones_layout)
        
        # Sección derecha
        self.derecha = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Imágenes
        self.imagen1 = Image(size_hint=(1, 0.3))
        self.imagen2 = Image(size_hint=(1, 0.3))

        self.boton_pequeno2 = Button(text=f'Guardar Imagen', background_color=(1, 0, 0, 1), size_hint=(0.5, 0.1))
        self.boton_pequeno2.bind(on_press=self.guardar_imagen)

        # Cuadros de texto
        self.texto1 = Label(text='Brillo: ', size_hint=(1, 0.1), color=(0.5, 0, 0.5, 1))
        self.texto2 = Label(text='Contraste: ', size_hint=(1, 0.1), color=(0.5, 0, 0.5, 1))

        # Agregar widgets
        self.derecha.add_widget(self.boton_pequeno2)
        self.derecha.add_widget(self.imagen1)
        self.derecha.add_widget(self.imagen2)
        self.derecha.add_widget(self.texto1)
        self.derecha.add_widget(self.texto2)

        # Agregar secciones a la ventana principal
        self.add_widget(self.izquierda)
        self.add_widget(self.derecha)
    
    def cargar_imagen(self, instance):
        filechooser.open_file(on_selection=self.selection)
    
    def selection(self, selection):
        self.imagenCV = cv2.imread(selection[0])
        buffer = cv2.flip(self.imagenCV, 0).tobytes()
        textura = Texture.create(size=(self.imagenCV.shape[1], self.imagenCV.shape[0]), colorfmt='bgr')
        textura.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.imagen_grande.texture = textura
    
    def calcular_brillo_contraste(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bright = np.mean(gray)
        contrast = np.std(gray)
        print(f'Brillo: {bright:.2f}')
        print(f'Contraste: {contrast:.2f}')
    
    def actualizar_imagen1(self, image):
        buffer = cv2.flip(image, 0).tobytes()
        textura = Texture.create(size=(image.shape[1], image.shape[0]), colorfmt='bgr')
        textura.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.imagen1.texture = textura
        self.calcular_brillo_contraste(image)

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
    
    def realizar_multiplicacion(self, instance):
        cte = 0.5
        imagePro = cv2.multiply(self.imagenCV, np.array([cte]))
        self.actualizar_imagen1(imagePro)
    
    def realizar_negativo(self, instance):
        L = 255
        imagePro = L - self.imagenCV
        self.actualizar_imagen1(imagePro)
    
    def realizar_gauss(self, instance):
        imagePro = cv2.GaussianBlur(self.imagenCV, (5, 5), 0)
        self.actualizar_imagen1(imagePro)
    
    def realizar_pasaAlto(self, instance):
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        imagePro = cv2.filter2D(self.imagenCV, -1, kernel)
        self.actualizar_imagen1(imagePro)

class myApp(App):
    def build(self):
        return miVentana()

if __name__ == '__main__':
    myApp().run()
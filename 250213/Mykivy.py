from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
from plyer import filechooser

class miVentana(BoxLayout): #herencia, definir mi ventana a partir de la libreria
    def __init__(self,**kwargs): #CONSTRUCTOR DE LA CLASE
        super().__init__(**kwargs) #CONSTRUCTOR ORIGINAL
        self.cajaInterna = BoxLayout(orientation = 'horizontal')
        self.orientation = 'vertical'
        self.L = Label(text = 'Hi', color = [0.2,0.2,1,1], bold = True, font_size = 50, size_hint=(0.7,0.3))
        self.cargar = Button(text = 'Cargar imagen', size_hint = (0.5,0.3), pos_hint = {'center_x':0.5, 'center_y':0.5})
        self.add_widget(self.cargar)

        self.cargar.bind(on_press = self.cargar_imagen) #vincular accion sobre boton
        self.cajaInterna.add_widget(self.L)
        self.image = Image(size_hint=(0.7,1))
        self.cajaInterna.add_widget(self.image)
        self.add_widget(self.cajaInterna)
        

    def cargar_imagen(self,instance):
        filechooser.open_file(on_selection = self.selection)
    
    def selection(self, selection):
        self.L.text = 'Auch!'
        #self.image.source = (r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\1-037.JPG")
        #imagenCV = cv2.imread(r"C:\Users\karen\OneDrive\Escritorio\DIP\250206\1-037.JPG")
        imagenCV = cv2.imread(selection[0])
        buffer = cv2.flip(imagenCV, 0).tostring() #convierte imagen a vector tipo texto
        textura = Texture.create(size = (imagenCV.shape[1], imagenCV.shape[0]), colorfmt = 'bgr')
        textura.blit_buffer(buffer, colorfmt = 'bgr', bufferfmt = 'ubyte')
        self.image.texture = textura

        
class myApp(App):
    def build(self):
        return miVentana()
    
if __name__ == '__main__':
    myApp().run()

#Los layouts nos ayudan a organizar las funciones en nuestra interfaz

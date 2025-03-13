from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
from plyer import filechooser
from kivy.clock import Clock

class miVentana(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        
        # Label
        self.L = Label(text='Hi', color=[0.2, 0.2, 1, 1], bold=True, font_size=50, size_hint=(0.7, 0.3))
        
        # Image display
        self.image = Image(size_hint=(0.7, 1))
        
        # Buttons
        self.cargar = Button(text='Cargar imagen', size_hint=(0.5, 0.3), pos_hint={'center_x': 0.5, 'center_y': 0.5})
        self.cargar.bind(on_press=self.cargar_imagen)
        
        self.captura = Button(text='Capture Camera', font_size=12, background_color=[0.5, 1, 0.5, 1], size_hint=(0.3, 0.4))
        self.captura.bind(on_press=self.iniciar_captura)
        
        self.guardar = Button(text='Save Capture', font_size=12, background_color=[1, 0.5, 0.5, 1], size_hint=(0.3, 0.4))
        self.guardar.bind(on_press=self.guardar_imagen)
        
        self.detener = Button(text='Stop Capture', font_size=12, background_color=[1, 0, 0, 1], size_hint=(0.3, 0.4))
        self.detener.bind(on_press=self.detener_captura)
        
        # Layouts
        self.cajaInterna = BoxLayout(orientation='horizontal')
        self.cajaInterna.add_widget(self.L)
        self.cajaInterna.add_widget(self.image)
        
        self.caja_h_2 = BoxLayout(orientation='horizontal', spacing=30)
        self.caja_h_2.add_widget(self.cargar)
        self.caja_h_2.add_widget(self.captura)
        self.caja_h_2.add_widget(self.guardar)
        self.caja_h_2.add_widget(self.detener)
        
        self.add_widget(self.cajaInterna)
        self.add_widget(self.caja_h_2)
        
        self.conteo = 0
        self.camera = None
    
    def cargar_imagen(self, instance):
        filechooser.open_file(on_selection=self.selection)
    
    def selection(self, selection):
        self.L.text = 'Auch!'
        imagenCV = cv2.imread(selection[0])
        buffer = cv2.flip(imagenCV, 0).tobytes()
        textura = Texture.create(size=(imagenCV.shape[1], imagenCV.shape[0]), colorfmt='bgr')
        textura.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = textura
    
    def iniciar_captura(self, instance):
        self.camera = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_image, 1.0/30.0)
    
    def update_image(self, dt):
        ret, imagenCV2 = self.camera.read()
        if ret:
            buffer = cv2.flip(imagenCV2, 0).tobytes()
            textura = Texture.create(size=(imagenCV2.shape[1], imagenCV2.shape[0]), colorfmt='bgr')
            textura.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = textura
    
    def guardar_imagen(self, instance):
        filechooser.save_file(on_selection=self.save_selection)
    
    def save_selection(self, selection):
        if selection:
            ret, imagenCV2 = self.camera.read()
            if ret:
                print(selection[0])
                cv2.imwrite(selection[0], imagenCV2)

    
    def detener_captura(self, instance):
        Clock.unschedule(self.update_image)
        if self.camera:
            self.camera.release()
            self.camera = None
        
class myApp(App):
    def build(self):
        return miVentana()
    
if __name__ == '__main__':
    myApp().run()
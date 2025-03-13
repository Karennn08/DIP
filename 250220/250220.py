from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
import cv2
from kivy.clock import Clock

class Miventana(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        
        self.L = Label(text='Detección de Rostros', color=[1, 0.2, 0.2, 1], bold=True, font_size=30)
        self.L2 = Label(text='Ojos detectados: 0', color=[0, 0.5, 1, 1], font_size=30)
        self.L3 = Label(text='Sonrisas detectadas: 0', color=[1, 1, 0, 1], font_size=30)
        
        self.cargar = Button(text="Activar Cámara y Detectar Rostros",
                             size_hint=(0.5, 0.3),
                             pos_hint={"center_x": 0.5, "center_y": 0.5})
        self.cargar.bind(on_press=self.activar_camara)
        
        self.imagen = Image(size_hint=(1, 2))  # Hace la caja del video más grande
        
        self.add_widget(self.L)
        self.add_widget(self.L2)
        self.add_widget(self.L3)
        self.add_widget(self.imagen)
        self.add_widget(self.cargar)
        
        self.captura = None
        
        # Cargar clasificadores preentrenados de OpenCV
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    def activar_camara(self, instance):
        if self.captura is None:
            self.captura = cv2.VideoCapture(0)
            Clock.schedule_interval(self.actualizar_frame, 1.0 / 30.0)
    
    def actualizar_frame(self, dt):
        ret, frame = self.captura.read()
        if ret:
            frame = cv2.flip(frame, 1)  # Corrige la orientación horizontal
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
            
            total_eyes = 0
            total_smiles = 0
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
                total_eyes += len(eyes)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                smiles = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))
                total_smiles += len(smiles)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)
            
            buffer = cv2.flip(frame, 0).tobytes()
            textura = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            textura.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.imagen.texture = textura
            
            self.L.text = f"Rostros detectados: {len(faces)}"
            self.L2.text = f"Ojos detectados: {total_eyes}"
            self.L3.text = f"Sonrisas detectadas: {total_smiles}"
    
    def on_stop(self):
        if self.captura:
            self.captura.release()
        
class MyApp(App):
    def build(self):
        return Miventana()

if __name__ == '__main__':
    MyApp().run()

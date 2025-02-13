#METODOS:
#constructor: funcion que se llama cuando se inicializa la clase
#Abstraccion, herencia, polimorfismo, encapsulamiento

class Persona:
    def __init__(self, nombre,apellido, edad): #constructor, self hace referencia a la instancia actual
        self.nombre = nombre
        self.apellido = apellido
        self.edad = edad

    def saludar(self): #self es el primer paramtro de toda funcion
        print(f"Hola, mi nombre es {self.nombre} {self.apellido} y tengo {self.edad} años.")

#Crear una instancia
persona1 = Persona('Mariana','Nieto', 21)
persona2 = Persona('Karen','Arango', 20)
persona1.saludar()
persona2.saludar()

#INTERFACES GRAFICAS: kivy o cualquier otro framework
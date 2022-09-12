#-------------------Librerias---------------------#
from tkinter import *
import os
import cv2
from matplotlib import pylot
from mtcnn.mtcnn import MTCNN 
import numpy as np 

#-------------------Pantalla principal-----------------------#

def Pantalla_principal():
    global pantalla #Variable globalizada para usarla en otras funciones
    pantalla = Tk()
    pantalla.geometry("300x250") #Tamaño de la ventana
    pantalla.title("Login visión artificial") #Título de pantalla
    Label(text= "Login inteligente", bg= "gray", width="300", height="2", font=("Verdana",13 )).pack()  #Características de la ventana



#---------------------Función registro de usuarios-------------------------------#
def registro(): #Variables globalizadas para poder llamarlas en otras funciones
    global usuario
    global contra
    global usuario_entrada
    global contra_entrada
    global pantalla_1
    pantalla_1 = Toplevel(pantalla) # Pantalla un nivel superios a la principal
    pantalla_1.title("Registro")
    pantalla_1.geometry("300x250") #Tamañod de ventana

#-------------------Función registro facial----------------------#
def registro_facial():
    #Capturamos el rostro
    cap = cv2.VideoCapture(0) #Elegimos la cámara con la que vamos a hacer la detencción
    while(True):
        ret, frame = cap.read() #Leemos el video
        cv2.imshow('Registro Facial', frame) #Mostramos el video en pantalla
        if cv2.waitKey(1) == 27:    #Cuando oprimamos "Escape" rompe el video
            break
    usuario_img = usuario.get()
    cv2.imwrite(usuario_img+".jpg",frame) #Guardamos la ultima captura del video como imagen y asignamos el nombre del usuario
    cap.release() #Cerramos
    cv2.destroyAllWindows()

    usuario_entrada.delete(0, END) #Limpiamos las variables de texto
    contra_entrada.delete(0, END)
    Label(pantalla_1, takefocus="Registro Facial Exitoso", fg="green", font=("Calibri",11)).pack()

#-----------------Función login----------------------------#
def login():
    global pantalla_2
    global verificacion_usuario
    global verificacion_contra
    global usuario_entrada2
    global contra_entrada2

    pantalla_2 = Toplevel(pantalla)
    pantalla_2.title("Login")
    pantalla_2.geometry("300x250") #Creamos la ventana
    Label(pantalla_2, text="Login facial: Debes ingresar un usuario").pack()
    Label(pantalla_2, text="Login tradicional: Debes ingresar usuario y contraseña").pack()
    Label(pantalla_2, text="").pack() #Espacio 

    verificacion_usuario = StringVar()
    verificacion_contrag = StringVar()
 
    #---------------Ingreso de datos--------------------------#
    Label(pantalla_2, text="Usuario: ").pack()
    usuario_entrada2 = Entry(pantalla_2, textvariable= verificacion_usuario)
    usuario_entrada2.pack()
    Label(pantalla_2, text="Contraseña: ").pack()
    contra_entrada2 = Entry(pantalla_2, textvariable= verificacion_contra)
    contra_entrada2.pack()
    Label(pantalla_2, takefocus="").pack()
    Button(pantalla_2, text="Inicio de Sesion Tradicional",width=20, height=1, command= verificaciom_login).pack()

    #---------------Botón login facial-----------------------#
    Label(pantalla_2, text="").pack()
    Button(pantalla_2, text="Inicio de sesión facial", width=20, height=1, command=login_facial).pack()

#----------------------Función login facial--------------------#
def login_facial():
    #Captura de rostro
    cap = cv2.VideoCapture(0)
    while(True):
        ret,frame = cap.read()
        cv2.imshow('Login Facial', frame)
        if cv.waitKey(1) == 27:
            break
    usuario_login = verificacion_usuario.get()
    cv2.imwrite(usuario_login+"LOG.jpg",frame)
    cap.release()
    cv2.destroyAllWindows()

    usuario_entrada2.delete(0, END)
    contra_entrada2.delete(0, END)

    def login_rostro(img, lista_resultados):
        data = pyplot.imread(img)
        for i in range (len(lista_resultados)):
            x1,y1,ancho,alto = lista_resultados[i]['box']
            x2,y2 = x1 + ancho, yi + alto
            pyplot.subplot(1, len(lista_resultados), i+1)
            pylot.axis('off')
            cara_registro = data[y1:y2, x1:x2]
            cara_registro = cv2.resize(cara_registro,(150,200), interpolation = cv2.INTER_CUBIC)
            cv2.imwrite(usuario_login+"LOG.jpg",cara_registro)
            return pyplot.imshow(data[y1:y2, x1:x2])
        pylot.show()

    #DETECTAMOS EL ROSTRO
    img = usuario_login+"LOG.jpg"
    pixeles = pyplot.imread(img)
    detector = MTCNN()
    caras = detector.detect_faces(pixeles)
    login_rostro(img,caras)

    def orb_sim(img1,img2):
        orb = cv2.ORB_create() #Creamos el objeto de comparación

        kpa, descriptor_A = orb_detecAndCompute(img1, None) #Creamer descriptor 1 y extraemos puntos claves
        kpa, descriptor_B = orb.orb_detecAndCompute(img2, None) #Creamos descriptor 2 y extraemos puntos claves

        comparador = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) #Creamos comparador de fuerza

        matches = comparador.match(descriptor_A, descriptor_B) #Aplicamos el comparador a los descriptores

        regiones_similares = (i for i in matches if i.distance < 70) #Extraemos las regiones similares en base a los puntos claves
        if len(matches) == 0:
            return 0
        return len(regiones_similares)/len(matches) #Exportamos el porcentaje de similitud
#-----------------Importamos las imagenes y llamamos la función comparación----------------#
image_archivos = os.listdir() #Vamos a importar la lista de archivos con la librería os
if usuario_login+".jpg" in image_archivos: #Comparamos los archivos con el que nos interesa
    rostro_registro = cv2.imread(usuario_login+".jpg",0) #Importamos el rostro del registro
    rostro_login = cv2.imread(usuario_login+"LOG.jpg",0) #Importamos el rostro del login
    similitud = orb_sim(rostro_registro, rostro_login)
    if similitud >= 0.9:
        Label(pantalla_2, text="Inicio de sesión exitoso", fg="green", font=("Calibri",11)).pack()
        print("Bienvenido al sistema usuario: "+usuario_login)
        print("Compatibilidad con la foto del registro: "+similitud)
    else:
        print("Rostro incorrecto, rectifique su usuario!")
        print("Compatibilidad con la foto del registro: "+similitud)
        Label(pantalla_2, text="Incompatibilidad de rostros", fg="red", font=("Calibri",11)).pack()
else:
    print("Usuario no encontrado")
    Label(pantalla_2, text="Usuario no encontrado", fg="red", font=("Calibri",11)).pack()

#-----------------Detección de rostro y exportación de pixeles---------------------------#
def registro_rostro(img, lista_resultados):
    data = pyplot.imread(img)
    for i in range (len(lista_resultados)):
        x1,y1,ancho,alto = lista_resultados[i]['box']
        x2,y2 = x1 + ancho, y1 + alto
        pyplot.subplot(1, len(lista_resultados), i+1)
        pyplot.axis('off')
        cara_registro = data[y1:y2, x1:x2]
        cara_registro = cv2.resize(cara_registro,(150, 200), interpolation = cv2.INTER_CUBIC) #GUARDAMOS LA IMAGEN CON UN TAMAÑO DE 150X200
        cv2.imwrite(usuario_img+".jpg", cara_registro)
        pylot.imshow(data[y1:y2, x1:x2])
    pylot.show()

img = usuario_img+".jpg"
pixeles = pyplot.imread(img)
detector = MTCNN()
caras = detector.detect_faces(pixeles)
registro_rostro(img,caras)

#---------------------Creación de entradas de datos------------------------------#
usuario = StringVar()
contra = StringVar()

Label(pantalla_1, text="Registro facial: Se debe asignar un usuario").pack()
Label(pantalla_1, text="Registro tradicional: Se debe asignar un usuario y contraseña").pack()
Label(pantalla_1, text=""),Pack()#Espacio
Label(pantalla_1, text="Usuario: ").pack() #Mostramos en pantalla 1 al usuario
usuario_entrada = Entry(pantalla_1, textvariable= usuario) #Variable de texto para que el usuario ingrese la información del usuario
usuario_entrada.pack()
Label(pantalla_1, text="Contraseña: ").pack() #Mostramos la contraseña en pantalla 1
contra_entrada = Entry(pantalla_1, textvariable= contra) #Variable de texto para que el usuario ingrese la información de la contraseña
contra_entrada.pack()
Label(pantalla_1, text="").pack()#Espacio para separar del botón
Button(pantalla_1, text="Registro Tradicional", width= 15, height= 1, command= registrar_usuario).pack() #Creamos el botón

                    #Botón para registro facial#
Label(pantalla_1, text="").pack()
Button(pantalla_1, takefocus="Registro Facial", width=15, height=1, command=registro_facial).pack()
#------------------Botones-------------------------#

Label(text="").pack() #Espacio entre el título y el primer botón
Button(text="Iniciar sesión", height="2", width="30", command= login).pack()
Label(text="").pack() #Espacio entre el primer y segund botón
Button(text="Registro de usuario", height="2", width="30", command= registro).pack()

pantalla.mainloop()

import cv2
import os
import numpy as np 

rutaDatos = r'C:\Users\USER\OneDrive\Documentos\Deteccion de mascarillas\DataSet'
direccion_lista = os.listdir(rutaDatos)
print("Lista archivos: ", direccion_lista)

labels = []
datosCaras = []
label = 0

for nombre_direccion in direccion_lista:
    direccion_ruta = rutaDatos + "/" + nombre_direccion

    for nombre_archivo in os.listdir(direccion_ruta):
        ruta_imagen = direccion_ruta + "/" + nombre_archivo
        print(ruta_imagen)
        imagen = cv2.imread(ruta_imagen, 0)
        cv2.imshow("Imagen", imagen) #Comentar despues de usar primera vez
        cv2.waitKey(10) #Comentar despues de usar primera vez

        datosCaras.append(imagen)
        labels.append(label)

    label += 1
print("Etiqueta 0: ", np.count_nonzero(np.array(labels)== 0))
print("Etiqueta 1: ", np.count_nonzero(np.array(labels)== 1))

#LBPH Reconocimiento facial
cara_tapabocas = cv2.face.LBPHFaceRecognizer_create()

#Entrenamiento
print("Iniciando entrenamiento...")
cara_tapabocas.train(datosCaras, np.array(labels))

#Almacenando datos (modelo)
cara_tapabocas.write("face_mask_model.xml")
print("Datos de entrenamiento almacenados")
#EJEMPLO VISIÓN ARTIFICIAL (RECONOCIMIENTO DE PATRONES)

import cv2
import numpy as np


tarjeta = cv2.imread('placaOriginal.png', cv2.IMREAD_GRAYSCALE)
plantilla = cv2.imread('circuito.png', cv2.IMREAD_GRAYSCALE)
alto, ancho = np.shape(plantilla)

resultado = cv2.matchTemplate(tarjeta, plantilla, cv2.TM_CCOEFF_NORMED)
min, max, pos_min, pos_max = cv2.minMaxLoc(resultado)

pixel_superior_izquierda = pos_max
pixel_inferior_derecha = (pos_max[0] + ancho, pos_max[0] + alto)

cv2.rectangle(tarjeta, pixel_superior_izquierda, pixel_inferior_derecha, 255, 4)

cv2.imshow('resultado', tarjeta)
cv2.waitKey(0)
cv2.destryAllWindows()
import cv2
import os
import mediapipe as mp 
mp_deteccion_rostros = mp.solutions.face_detection

LABELS = ["tapabocas, noTapabocas"]

#Leemos los datos del modelo entrenado
cara_tapabocas = cv2.face.LBPHFaceRecognizer_create
cara_tapabocas.read("face_mask_model.xml")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_deteccion_rostros.FaceDetecion(
    min_detection_confidence = 0.5) as face_detection:

    while True:
        ret, frame = cap.read()
        if ret == False: break
        frame = cv2.flip(frame, 1)

        height, width = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = face_detection.process(frame_rgb)

        if resultados.detections is not None:
            for detection in resultados.detections:
                xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                width = int(detection.location_data.relative_bounding_box.width * width)
                height = int(detection.location_data.relative_bounding_box.height * height)
                if xmin < 0 and ymin < 0:
                    continue

                cv2.rectangle(frame , (xmin,ymin), (xmin + width, ymin + height), (0, 255, 0), 5)

                imagen_cara = frame[ymin : ymin + height, xmin : xmin + width]
                imagen_cara = cv2.cvtColor(imagen_cara, cv2.COLOR_BGR2GRAY)
                imagen_cara = cv2.resize(imagen_cara, (72, 72), INTERPOLATION = cv2.INTER_CUBIC)
                
                resultado = cara_tapabocas.predict(imagen_cara)
                #cv2.putText(frame, "{}".format(result), (xmin, ymin - 5), 1, 1.3, (210, 124, 176), 1, cv2.LINE_AA)

                if resultado[1] < 150:
                    color = (0, 255, 0) if LABELS[resultado[0]] == "tapabocas" else (0, 0, 255)
                    cv2.putText(frame, "{}".format(LABELS[resultado[0]]), (xmin, ymin - 15), 2, 1, color, 1, cv2.LINE_AA)
                    cv2.rectangle(frame, (xmin, ymin), (xmin +width, ymin + height), color, 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break 

cap.release()
cv2.destroyAllWindows()
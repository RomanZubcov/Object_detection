# -*- coding: utf-8 -*-

import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Încarcă modelul antrenat
model = tf.keras.models.load_model('/home/roman/Desktop/proiect/keras_model.h5')

# Etichetele claselor
class_labels = open("/home/roman/Desktop/proiect/labels.txt", "r").readlines()

# Inițializează cameră web
cap = cv2.VideoCapture(0)

while True:
    # Citește frame-ul de la cameră
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Redimensionează frame-ul la dimensiunea modelului
    input_image = cv2.resize(frame, (224, 224))
    
    # Preprocesează imaginea
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    
    # Realizează predicția folosind modelul
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    
    # Obține coordonatele dreptunghiului pentru obiectul detectat
    x, y, w, h = cv2.boundingRect(frame)
    
    # Desenează dreptunghiul în jurul obiectului detectat
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Desenează eticheta clasei pe frame
    cv2.putText(frame, predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Afișează frame-ul rezultat
    cv2.imshow('Object Detection', frame)
    
    # Verifică apăsarea tastei 'q' pentru a ieși din bucla while
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Eliberează resursele
cap.release()
cv2.destroyAllWindows()

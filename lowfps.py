import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Încarcă modelul antrenat
model = tf.keras.models.load_model('keras_model.h5')

# Etichetele claselor
class_labels = open("labels.txt", "r").readlines()

# Inițializează cameră web
cap = cv2.VideoCapture(0)

while True:
    # Citește frame-ul de la cameră
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Aplică algoritmul de detectare a obiectelor
    bbox, label, conf = cv.detect_common_objects(frame)
    
    # Desenează dreptunghiurile delimitatoare în jurul obiectelor detectate
    output_img = draw_bbox(frame, bbox, label, conf)
    
    # Redimensionează frame-ul la dimensiunea modelului pentru recunoașterea obiectelor
    input_image = cv2.resize(frame, (224, 224))
    
    # Preprocesează imaginea
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    
    # Realizează predicția folosind modelul pentru recunoașterea obiectelor
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    
    # Desenează eticheta clasei pe frame
    cv2.putText(output_img, predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Afișează frame-ul rezultat
    cv2.imshow('Object Detection and Counting', output_img)
    
    # Verifică apăsarea tastei 'q' pentru a ieși din bucla while
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Eliberează resursele
cap.release()
cv2.destroyAllWindows()

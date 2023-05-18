# -*- coding: utf-8 -*-

import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Load
model = tf.keras.models.load_model('/home/roman/Desktop/proiect/keras_model.h5')

# Class labels
class_labels = open("/home/roman/Desktop/proiect/labels.txt", "r").readlines()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define the colors for the frames
frame_colors = {'cola': (0, 0, 255), 'sprite': (0, 255, 0), 'fanta': (255, 0, 0)}

# Define the font settings for the labels
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_thickness = 2

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Resize the frame to match the model's input size
    input_image = cv2.resize(frame, (224, 224))
    
    # Preprocess the image
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)
    
    # Make predictions using the model
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index].strip()
    
    # Get the color for the frame based on the predicted class label
    frame_color = frame_colors.get(predicted_class_label, (255, 255, 255))
    
    # Find contours of the detected objects
    contours, _ = cv2.findContours(cv2.Canny(frame, 100, 200), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw a rectangle around each detected object
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), frame_color, 2)
    
    # Draw the class label on the frame
    label_size, _ = cv2.getTextSize(predicted_class_label, font, font_scale, font_thickness)
    label_x = int((frame.shape[1] - label_size[0]) / 2)
    label_y = int((frame.shape[0] + label_size[1]) / 2)
    cv2.putText(frame, predicted_class_label, (label_x, label_y), font, font_scale, frame_color, font_thickness, cv2.LINE_AA)
    
    # Display the resulting frame
    cv2.imshow('Object Detection', frame)
    
    # Check if 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

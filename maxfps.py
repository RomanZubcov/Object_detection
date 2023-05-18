import cv2
import tensorflow as tf
import numpy as np
from tensorflow import keras

# Load the trained model
model = tf.keras.models.load_model('keras_model.h5')

# Class labels
class_labels = open("labels.txt", "r").readlines()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Create a window for the live stream
cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)

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
    
    # Perform inference using the model
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    
    # Draw the class label on the frame
    cv2.putText(frame, predicted_class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Show the resulting frame
    cv2.imshow('Object Detection', frame)
    
    # Check if 'q' key is pressed to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

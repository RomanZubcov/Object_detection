import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the Keras model (.h5)
keras_model = load_model('/home/roman/Desktop/proiect/keras_model.h5')

# Convert the Keras model to TensorFlow SavedModel format
tf.saved_model.save(keras_model, '/home/roman/Desktop/proiect/savedmodel')

# Convert the SavedModel to a frozen graph
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('/home/roman/Desktop/proiect/savedmodel')
converter.optimizations = [tf.compat.v1.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the frozen graph (.pb)
with tf.io.gfile.GFile('/home/roman/Desktop/proiect/graph/frozen_graph.pb', 'wb') as f:
    f.write(tflite_model)

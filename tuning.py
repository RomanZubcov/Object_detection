import tensorflow as tf

# Setările fine-tuning
base_model_path = '/home/roman/Desktop/proiect/keras_model.h5'  # Calea către modelul preantrenat
train_data_dir = '/home/roman/Desktop/proiect/train_data'  # Calea către directorul cu datele de antrenament
num_classes = 3  # Numărul de clase din setul de date
epochs = 100  # Numărul de epoci pentru antrenament
batch_size = 64  # Dimensiunea lotului pentru antrenament

# Încărcați modelul preantrenat
base_model = tf.keras.models.load_model(base_model_path)

# Congelați straturile de bază
for layer in base_model.layers:
    layer.trainable = False

# Adăugați straturile personalizate
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compilați modelul
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Preprocesare date de antrenament
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,  # Rescalare valorilor pixelilor în intervalul [0, 1]
    rotation_range=20,  # Unghiuri de rotație aleatorii în intervalul [-20, 20]
    width_shift_range=0.2,  # Deplasare orizontală aleatorie a imaginilor
    height_shift_range=0.2,  # Deplasare verticală aleatorie a imaginilor
    shear_range=0.2,  # Deformare în plan a imaginilor
    zoom_range=0.2,  # Mărire/împuținare aleatoare a imaginilor
    horizontal_flip=True,  # Răsucire orizontală aleatoare a imaginilor
    fill_mode='nearest'  # Umplerea pixelilor din zonele nou-create
)

# Încărcați datele de antrenament
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

# Antrenarea modelului
model.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=train_generator.n // batch_size
)

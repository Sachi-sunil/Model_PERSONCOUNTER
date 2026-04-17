import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Image parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 8

# Load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "data/cnn/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "data/cnn/test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# Normalize
normalization_layer = layers.Rescaling(1./255)
train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

# Build CNN model
model = models.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # binary classification
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Save model
model.save("models/cnn_model.h5")

print("Model trained and saved.")

import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model("models/cnn_model.h5")

IMG_SIZE = 128

# Load image
image = cv2.imread("test.jpg")
image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image_array = image_resized / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Predict
prediction = model.predict(image_array)

if prediction[0][0] > 0.5:
    print("High Crowd (Possible Overload)")
else:
    print("Low Crowd (Safe)")
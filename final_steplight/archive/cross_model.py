import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2 as cv

# Load your specific images
cross_image = cv.imread('cross/cross.png')
no_cross_image = cv.imread('no_cross/no_cross.png')

# Preprocess images
def preprocess_image(image, image_size=(64, 64)):
    image = cv.resize(image, image_size)
    image = image.astype('float32') / 255.0
    return image

cross_image = preprocess_image(cross_image)
no_cross_image = preprocess_image(no_cross_image)

cross_label = np.array([1])
no_cross_label = np.array([0])

# Combine the two images
images = np.array([cross_image, no_cross_image])
labels = np.array([cross_label, no_cross_label])

# Create a very aggressive data augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=1,  # Minimal rotation
    width_shift_range=0.01,  # Minimal width shift
    height_shift_range=0.01,  # Minimal height shift
    shear_range=0.01,  # Minimal shear
    zoom_range=0.01,  # Minimal zoom
    horizontal_flip=False,  # Disable horizontal flip
    fill_mode='nearest'
)

# Create a very simple model with dropout to prevent complete overfitting
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),  # Light dropout
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate augmented versions of these specific images
augmented_images = []
augmented_labels = []

# Generate exactly 1000 augmented versions
for _ in range(500):  # This will generate ~1000 images
    for x, y in datagen.flow(images, labels, batch_size=2):
        augmented_images.append(x[0])
        augmented_images.append(x[1])
        augmented_labels.append(y[0])
        augmented_labels.append(y[1])
        if len(augmented_images) >= 1000:
            break

augmented_images = np.array(augmented_images)
augmented_labels = np.array(augmented_labels)

# Train for many epochs to memorize these specific images
model.fit(augmented_images, augmented_labels, epochs=200, batch_size=32)

# Save the model
model.save('exact_image_classifier.keras')

# Optional: Verify prediction on original images
print("Cross Image Prediction:", model.predict(np.array([cross_image]))[0][0])
print("No Cross Image Prediction:", model.predict(np.array([no_cross_image]))[0][0])
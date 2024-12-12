import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Constants
IMG_SIZE = 224  # Standard input size for MobileNetV2
BATCH_SIZE = 8  # Batch size for training
EPOCHS = 20  # Number of training epochs
CLASS_NAMES = ['cross', 'no_cross']  # Class labels

# Creates a data augmentation layer.
# Returns:
#   A Sequential model for image augmentation.
def create_data_augmentation():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.2),
    ])

# Loads and preprocesses the image dataset.
# Args:
#   cross_dir: Directory containing 'cross' images.
#   no_cross_dir: Directory containing 'no_cross' images.
# Returns:
#   Tuple of numpy arrays (images, labels).
def load_and_preprocess_data(cross_dir, no_cross_dir):
    images = []
    labels = []

    # Load 'cross' images
    cross_count = 0
    for img_path in os.listdir(cross_dir):
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(cross_dir, img_path), target_size=(IMG_SIZE, IMG_SIZE)
            )
            images.append(tf.keras.preprocessing.image.img_to_array(img))
            labels.append(1)  # Label 1 for 'cross'
            cross_count += 1

    # Load 'no_cross' images
    no_cross_count = 0
    for img_path in os.listdir(no_cross_dir):
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(no_cross_dir, img_path), target_size=(IMG_SIZE, IMG_SIZE)
            )
            images.append(tf.keras.preprocessing.image.img_to_array(img))
            labels.append(0)  # Label 0 for 'no_cross'
            no_cross_count += 1

    # Print dataset composition
    print(f"\nDataset composition:")
    print(f"Cross images: {cross_count}")
    print(f"No cross images: {no_cross_count}")
    print(f"Total images: {cross_count + no_cross_count}")

    # Convert to numpy arrays and normalize
    images = np.array(images) / 255.0
    labels = np.array(labels)

    return images, labels

# Creates the MobileNetV2-based binary classification model.
# Returns:
#   A compiled Keras model.
def create_model():
    # Load MobileNetV2 as the base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Fine-tune the last 40 layers
    for layer in base_model.layers[-40:]:
        layer.trainable = True

    # Build the full model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        create_data_augmentation(),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])

    return model

# Main function to train, evaluate, and convert the model to TFLite format.
def main():
    cross_dir = 'cross'  # Path to 'cross' images
    no_cross_dir = 'no_cross'  # Path to 'no_cross' images

    # Load and preprocess data
    print("Loading and preprocessing data...")
    images, labels = load_and_preprocess_data(cross_dir, no_cross_dir)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.20, random_state=42, stratify=labels
    )
    print("\nData split sizes:")
    print(f"Training set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    print("\nTraining set composition:", Counter(y_train))
    print("Test set composition:", Counter(y_test))

    # Create and compile the model
    print("\nCreating model...")
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )

    # Evaluate the model on the test set
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"\nBest validation accuracy: {max(history.history['val_accuracy']):.4f}")

    # Convert the trained model to TFLite format
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('crosswalk_detector.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved as 'crosswalk_detector.tflite'")


if __name__ == "__main__":
    main()
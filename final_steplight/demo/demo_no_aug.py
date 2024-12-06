import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import time

CLASS_NAMES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

def prepare_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values to be between 0 and 1
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

def build_model():
    inputs = layers.Input(shape=(32, 32, 3))

    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=30, batch_size=64):
    print("Starting training...")
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        batch_size=batch_size,
        verbose=1
    )

    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    return history

def evaluate_model(model, x_test, y_test):
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f"\nTest accuracy: {test_accuracy * 100:.2f}%")

def predict_sample(model, x_test, y_test, sample_index=42):
    sample_image = x_test[sample_index]
    true_label = np.argmax(y_test[sample_index])
    sample_input = np.expand_dims(sample_image, axis=0)

    prediction = model.predict(sample_input)
    predicted_class = np.argmax(prediction)

    print("\nSample Prediction:")
    print(f"True Label: {CLASS_NAMES[true_label]}")
    print(f"Predicted Label: {CLASS_NAMES[predicted_class]}")
    print("Prediction Probabilities:")
    for i, prob in enumerate(prediction[0]):
        print(f"{CLASS_NAMES[i]}: {prob * 100:.2f}%")

def main():
    x_train, y_train, x_test, y_test = prepare_data()

    # Build the model
    model = build_model()

    history = train_model(model, x_train, y_train, x_test, y_test, epochs=30)
    evaluate_model(model, x_test, y_test)
    predict_sample(model, x_test, y_test)
    model.save('cifar10_model.keras')

if __name__ == '__main__':
    main()
import tensorflow as tf
from tensorflow.keras import *

# Load the MNIST dataset
# Splits the dataset into training and testing sets and normalizes the pixel values to the range [0, 1]
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the model architecture
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images into 1D vectors
    layers.Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    layers.Dense(10)  # Output layer with 10 neurons
])

# Compile the model
model.compile(
    optimizer=optimizers.Adam(0.001),  # Optimizer with a learning rate of 0.001
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),  # Loss function for logits output
    metrics=[metrics.SparseCategoricalAccuracy()],  # Metric to evaluate accuracy
)

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')  # Prints the test accuracy.

# Save the trained model to a file
model.save('mnist_model.keras')
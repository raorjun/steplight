import numpy as np
import tflite_runtime.interpreter as tflite
import cv2

# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path='mnist_model.tflite')
interpreter.allocate_tensors()

# Retrieve input and output tensor details from the model.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Preprocesses an input image for model inference.
# Args:
#   image: The input image in BGR format.
# Returns:
#   A grayscale, resized, and normalized image suitable for the model.
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale.
    image = cv2.resize(image, (28, 28))  # Resize to match model input dimensions.
    image = image / 255.0  # Normalize pixel values to the range [0, 1].
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dimension.
    return image

# Initialize the webcam feed.
cap = cv2.VideoCapture(0)

# Main loop to process webcam frames.
while True:
    ret, frame = cap.read()  # Capture a frame from the webcam.
    if not ret:
        break  # Exit the loop if the frame is not captured successfully.

    # Preprocess the captured frame for inference.
    input_data = preprocess_image(frame)

    # Perform inference with the TFLite model.
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Interpret the model output and generate a prediction text.
    if output_data.size == 0 or np.max(output_data) < 0.5:
        predicted_text = 'None'  # No confident prediction.
    else:
        predicted_digit = np.argmax(output_data)  # Find the digit with the highest probability.
        predicted_text = f'Predicted: {predicted_digit}'

    # Overlay the prediction text on the webcam frame.
    cv2.putText(frame, predicted_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame in a window.
    cv2.imshow('Webcam Feed', frame)

    # Exit the loop if the 'q' key is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and close the display window.
cap.release()
cv2.destroyAllWindows()

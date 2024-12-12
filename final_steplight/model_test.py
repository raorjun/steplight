import cv2
import numpy as np
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.adapters.tensor_image import TensorImage

# Load the TFLite model and allocate tensors
interpreter = make_interpreter('crosswalk_detector_edgetpu.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_frame(frame, img_size=224):
    """Resize and normalize the frame for the model."""
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame.astype('float32') / 255.0
    frame = np.expand_dims(frame, axis=0)
    return frame

def predict(frame):
    """Run inference and return the prediction."""
    tensor_image = TensorImage(frame)
    interpreter.set_tensor(input_details[0]['index'], tensor_image.tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]

# Initialize webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Prediction parameters
CROSS_THRESHOLD = 0.85  # Threshold for "cross"
CONFIDENCE_THRESHOLD = 0.75  # Minimum confidence required

while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess the frame and predict
    preprocessed_frame = preprocess_frame(frame)
    prediction = predict(preprocessed_frame)
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Determine label and color
    if prediction > CROSS_THRESHOLD and confidence > CONFIDENCE_THRESHOLD:
        label = "Cross"
        color = (0, 255, 0)  # Green
    else:
        label = "No Cross"
        color = (0, 0, 255)  # Red

    # Display label and confidence
    cv2.putText(frame, f"{label}: {confidence:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display threshold
    cv2.putText(frame, f"Thresh: {CROSS_THRESHOLD}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow('Crosswalk Detector', frame)

    # Key controls for thresholds and exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit
        break
    elif key == ord('+') and CROSS_THRESHOLD < 0.95:  # Increase threshold
        CROSS_THRESHOLD += 0.05
    elif key == ord('-') and CROSS_THRESHOLD > 0.55:  # Decrease threshold
        CROSS_THRESHOLD -= 0.05

# Release resources
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Constants matching your training code
IMG_SIZE = 224
CLASS_NAMES = ['no_cross', 'cross']  # Note: 0=no_cross, 1=cross from your training

# Load the TFLite model
interpreter = tflite.Interpreter(model_path='crosswalk_detector.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_frame(frame):
    """Preprocess frame exactly as in training"""
    # Resize to match MobileNetV2 input size
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    # Normalize to [0,1] as done in training
    frame = frame.astype('float32') / 255.0
    # Add batch dimension
    frame = np.expand_dims(frame, axis=0)
    return frame

# Initialize camera
cap = cv2.VideoCapture(0)  # Try 2 or other numbers if this doesn't work
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Press 'q' to quit")
print("Press '+'/'-' to adjust threshold")

# Threshold for classification
THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Keep original frame for display
    display_frame = frame.copy()

    # Preprocess frame
    processed_frame = preprocess_frame(frame)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], processed_frame)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Determine class and color
    is_cross = prediction > THRESHOLD
    label = CLASS_NAMES[1] if is_cross else CLASS_NAMES[0]
    confidence = prediction if is_cross else (1 - prediction)
    color = (0, 255, 0) if is_cross else (0, 0, 255)  # Green for cross, Red for no_cross

    # Draw predictions on frame
    cv2.putText(display_frame, f"{label}: {confidence:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(display_frame, f"Threshold: {THRESHOLD:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Show the frame
    cv2.imshow('Crosswalk Detector', display_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') and THRESHOLD < 0.95:
        THRESHOLD += 0.05
    elif key == ord('-') and THRESHOLD > 0.05:
        THRESHOLD -= 0.05

# Cleanup
cap.release()
cv2.destroyAllWindows()
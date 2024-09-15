import cv2
import numpy as np
from openvino.runtime import Core, Tensor

# Initialize OpenVINO Runtime
core = Core()

# Load and compile the face detection model
face_model_path = "models/face-detection-adas-0001/face-detection-adas-0001.xml"
face_model = core.read_model(face_model_path)
face_compiled_model = core.compile_model(face_model, "CPU")

# Load and compile the emotion recognition model
emotion_model_path = "models/emotions-recognition-retail-0003/emotions-recognition-retail-0003.xml"
emotion_model = core.read_model(emotion_model_path)
emotion_compiled_model = core.compile_model(emotion_model, "CPU")

# Get input size for the models
def get_input_size(model):
    input_tensor = model.input(0)  # Get the first input tensor
    input_shape = input_tensor.shape
    return (input_shape[3], input_shape[2])  # (width, height)

face_input_size = get_input_size(face_model)
emotion_input_size = get_input_size(emotion_model)

print(f"Face model input size: {face_input_size}")
print(f"Emotion model input size: {emotion_input_size}")

# Preprocess the image
def preprocess_face(image, input_size):
    resized_image = cv2.resize(image, (input_size[0], input_size[1]))
    resized_image = resized_image.astype(np.float32)  # Convert to float32
    resized_image = resized_image.transpose((2, 0, 1))  # Change to CHW format
    resized_image = resized_image[None, :]  # Add batch dimension
    return resized_image

# Post-process the output to extract faces
def postprocess_face(output, threshold=0.5):
    boxes = []
    for detection in output[0][0]:  # Loop through all detections
        if detection[2] > threshold:  # Confidence threshold
            xmin = int(detection[3] * image.shape[1])
            ymin = int(detection[4] * image.shape[0])
            xmax = int(detection[5] * image.shape[1])
            ymax = int(detection[6] * image.shape[0])
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes

# Preprocess the face region for emotion recognition
def preprocess_emotion(face_image, input_size):
    resized_face = cv2.resize(face_image, (input_size[0], input_size[1]))
    resized_face = resized_face.astype(np.float32)  # Convert to float32
    resized_face = resized_face.transpose((2, 0, 1))  # Change to CHW format
    resized_face = resized_face[None, :]  # Add batch dimension
    return resized_face

# Post-process the emotion output
def postprocess_emotion(output):
    emotions = ["neutral", "happiness", "surprise", "anger", "sadness"]
    emotion_scores = output[0][0]  # Assuming output shape is [1, 5, 1, 1]
    emotion_index = np.argmax(emotion_scores)
    return emotions[emotion_index]

# Load the image
image_path = "aa.png"
image = cv2.imread(image_path)
original_image = image.copy()

# Preprocess the image for face detection
preprocessed_image = preprocess_face(image, face_input_size)

# Convert NumPy array to OpenVINO Tensor
input_tensor = Tensor(preprocessed_image)

# Perform face detection inference
face_infer_request = face_compiled_model.create_infer_request()
face_infer_request.set_input_tensor(input_tensor)
face_infer_request.infer()
face_output = face_infer_request.get_output_tensor().data

# Post-process the face detection output
detected_faces = postprocess_face(face_output)

# Process each detected face for emotion recognition
for (xmin, ymin, xmax, ymax) in detected_faces:
    face_region = original_image[ymin:ymax, xmin:xmax]
    
    # Preprocess the face region for emotion recognition
    preprocessed_face = preprocess_emotion(face_region, emotion_input_size)
    
    # Convert NumPy array to OpenVINO Tensor
    emotion_tensor = Tensor(preprocessed_face)
    
    # Perform emotion recognition inference
    emotion_infer_request = emotion_compiled_model.create_infer_request()
    emotion_infer_request.set_input_tensor(emotion_tensor)
    emotion_infer_request.infer()
    emotion_output = emotion_infer_request.get_output_tensor().data
    
    # Post-process the emotion recognition output
    emotion = postprocess_emotion(emotion_output)
    
    # Draw bounding boxes and emotion labels on the original image
    cv2.rectangle(original_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(original_image, emotion, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

# Display the result
cv2.imshow("Detected Faces and Emotions", original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

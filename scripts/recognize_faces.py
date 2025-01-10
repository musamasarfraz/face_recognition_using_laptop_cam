import cv2
import dlib
import numpy as np
import pickle
from scipy.spatial import distance

# Paths
model_path = "models/face_encodings.pkl"

# Load the trained face encodings and labels
with open(model_path, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_labels = data["labels"]

# Initialize dlib's face detector and shape predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# Start the webcam feed
cam = cv2.VideoCapture(0)

print("Starting live facial recognition. Press 'q' to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    faces = face_detector(rgb_frame, 0)

    for face in faces:
        # Get facial landmarks
        shape = shape_predictor(rgb_frame, face)

        # Compute the face embedding
        face_encoding = np.array(face_recognition_model.compute_face_descriptor(rgb_frame, shape))

        # Compare with known encodings
        distances = [distance.euclidean(face_encoding, known_encoding) for known_encoding in known_encodings]
        min_distance = min(distances) if distances else None

        # Threshold for recognition
        threshold = 0.6
        if min_distance is not None and min_distance < threshold:
            index = distances.index(min_distance)
            name = known_labels[index]
        else:
            name = "Unknown"

        # Draw a rectangle around the face and label it
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Live Facial Recognition", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()

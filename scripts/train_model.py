import os
import cv2
import dlib
import numpy as np
import pickle

# Paths
data_dir = "data"  # Directory containing face images
model_path = "models/face_encodings.pkl"  # Path to save encodings

# Initialize dlib's face detector, shape predictor, and recognition model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

# Initialize storage for encodings and labels
encodings = []
labels = []

# Iterate over each person's folder in the data directory
for person_name in os.listdir(data_dir):
    person_path = os.path.join(data_dir, person_name)
    
    if not os.path.isdir(person_path):
        continue
    
    print(f"Processing images for {person_name}...")
    
    # Iterate over each image in the person's folder
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        
        # Load the image
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping invalid image: {image_path}")
            continue

        # Debugging: Print image properties
        print(f"Processing {image_path}: Shape={image.shape}, Dtype={image.dtype}")

        # Check image type
        if image.dtype != np.uint8:
            print(f"Error: Unsupported image type in {image_path}. Expected uint8, got {image.dtype}")
            continue

        # Convert to RGB
        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error converting image to RGB for {image_path}: {e}")
            continue

        # Resize image to ensure compatibility
        resized_image = cv2.resize(rgb_image, (300, 300))

        # Try detecting faces using dlib
        try:
            faces = face_detector(resized_image, 0)
        except RuntimeError as e:
            print(f"Error using dlib detector for {image_path}: {e}")
            faces = []

        if not faces:
            print(f"No faces detected in {image_path}. Skipping...")
            continue

        # Process detected faces
        for face in faces:
            try:
                # Get facial landmarks
                shape = shape_predictor(resized_image, face)

                # Compute the face embedding
                face_encoding = np.array(face_recognition_model.compute_face_descriptor(resized_image, shape))

                # Store the encoding and label
                encodings.append(face_encoding)
                labels.append(person_name)
            except Exception as e:
                print(f"Error processing face in {image_path}: {e}")
                continue

# Save the encodings and labels to a file
data = {"encodings": encodings, "labels": labels}

with open(model_path, "wb") as f:
    pickle.dump(data, f)

print(f"Training complete. Encodings saved to {model_path}.")

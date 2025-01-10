import cv2
import os

# Directory to save captured faces
output_dir = "data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set width
cam.set(4, 480)  # Set height

# Load Haar cascade for face detection
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Input name for labeling
name = input("Enter the name of the person to capture faces: ").strip()
save_path = os.path.join(output_dir, name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Initialize counters
count = 0
max_images = 250  # Number of images to capture

print(f"Capturing images for {name}. Press 'q' to stop.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y + h, x:x + w]

        # Resize the face to 300x300
        resized_face = cv2.resize(face, (300, 300))

        # Save the detected face
        face_path = os.path.join(save_path, f"{count}.jpg")
        cv2.imwrite(face_path, resized_face)
        count += 1

        if count >= max_images:
            print(f"Collected {max_images} images for {name}.")
            cam.release()
            cv2.destroyAllWindows()
            exit()

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Face Capture", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()

print(f"Captured {count} images for {name}. Data saved in {save_path}.")

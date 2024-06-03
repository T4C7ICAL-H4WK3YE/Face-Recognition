import cv2
import face_recognition
import sqlite3
import numpy as np

# Database Connection
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

# Load known faces and encodings from database
known_face_encodings = []
known_face_names = []
cursor.execute("SELECT name, encoding FROM faces")
for row in cursor:
    name, encoding = row
    known_face_names.append(name)
    # Convert the encoding to a NumPy array
    known_face_encodings.append(np.frombuffer(encoding, dtype=np.float64))

# Function to detect faces in a frame
def detect_faces(frame):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Compare detected encodings to known faces
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Find closest match
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# Video Processing
# p="face1.mp4"
video_capture = cv2.VideoCapture(0)  # Replace with your video source
#video_capture = cv2.VideoCapture(1)  # Replace with your video source
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    detect_faces(frame)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Quit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()

conn.close()
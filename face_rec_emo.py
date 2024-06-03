import cv2
import sqlite3
import face_recognition
import numpy as np
from deepface import DeepFace as df
from datetime import datetime, timedelta
import pandas as pd
import os

# Check if the Excel file exists
if not os.path.exists('face_data_gender.xlsx'):
    # Create an empty DataFrame
    empty_df = pd.DataFrame(columns=['Name', 'Gender', 'Timestamp'])
    # Write the empty DataFrame to an Excel file
    empty_df.to_excel('face_data_gender.xlsx', index=False)

# Database connection
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

# Load known faces and encodings from the database
known_face_encodings = []
known_face_names = []

cursor.execute("SELECT name, encoding FROM faces")
for row in cursor:
    name, encoding = row
    known_face_names.append(name)
    known_face_encodings.append(np.frombuffer(encoding, dtype=np.float64))

# Initialize dictionary to store timestamps for recognized faces
recognized_face_info = {}
last_detection_time = {}

# Initialize the count variable
count = 0

# Function to detect faces in a frame
def detect_faces(frame):
    global recognized_face_info, last_detection_time, count

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    threshold = 0.4  # Set a threshold for face distance

    current_time = datetime.now()

    # Compare detected encodings to known faces
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=threshold)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Get or set information for recognized face
            if name not in recognized_face_info:
                recognized_face_info[name] = {'gender': '', 'emotion': '', 'timestamp': current_time}
                last_detection_time[name] = current_time
                count += 1
            else:
                if current_time - last_detection_time[name] > timedelta(seconds=7):
                    last_detection_time[name] = current_time
                    count += 1

            # Analyze face for emotion and gender
            analysis = df.analyze(frame, actions=['emotion', 'gender'], enforce_detection=False)
            recognized_face_info[name]['emotion'] = analysis[0]['dominant_emotion']
            recognized_face_info[name]['gender'] = analysis[0]['dominant_gender']

            # Display name, gender, emotion, and timestamp on the screen
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name if name != 'Unknown' else 'Unknown', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, recognized_face_info[name]['gender'], (left, top - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, recognized_face_info[name]['emotion'], (left, top - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, recognized_face_info[name]['timestamp'].strftime("%Y-%m-%d %H:%M:%S"), (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Update Excel file with recognized face info (if not 'Unknown')
            if name != 'Unknown':
                update_excel(name, recognized_face_info[name]['gender'], recognized_face_info[name]['timestamp'])
        else:
            name = 'Unknown'
            unknown_encoding_str = face_encoding.tostring()
            if name not in last_detection_time:
                last_detection_time[name] = current_time
                count += 1
            else:
                if current_time - last_detection_time[name] > timedelta(seconds=7):
                    last_detection_time[name] = current_time
                    count += 1

            # Display "Unknown"
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, 'Unknown', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


# Function to update Excel file with recognized face info
def update_excel(name, gender, timestamp):
    # Read existing data from Excel file
    existing_data = pd.read_excel('face_data_gender.xlsx')
    
    # Remove rows with a blank name column
    existing_data = existing_data.dropna(subset=['Name'])

    # Check if the name is not blank and not already in the Excel sheet
    if name.strip() and name not in existing_data['Name'].tolist():
        # Append data to existing
        dataframe = pd.DataFrame([[name, gender, timestamp]], columns=['Name', 'Gender', 'Timestamp'])
        updated_data = pd.concat([existing_data, dataframe], ignore_index=True)
        updated_data.to_excel('face_data_gender.xlsx', index=False)

# Video Processing
# Use cv2.CAP_DSHOW for DirectShow (Windows), or try other backends if necessary
video_capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set properties for the capture device (optional)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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
print(f"Total faces detected: {count}")
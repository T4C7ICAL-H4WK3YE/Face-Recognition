# import argparse
import logging
import os
import sys
import sqlite3
import time
import argparse
import torch
import cv2
import numpy as np
import dlib

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.config.fd_config import define_img_size

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Real-time Face Recognition')
parser.add_argument('--model_path', type=str, help='Path to the trained face detection model')
parser.add_argument('--landmarks_path', type=str, default='shape_predictor_5_face_landmarks.dat', help='Path to the dlib landmarks file')
args = parser.parse_args()

# Initialize logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define input image size
input_img_size = 320
define_img_size(input_img_size)
logging.info("Input size: {}".format(input_img_size))

# Set device (CUDA or CPU)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logging.info("Using device: {}".format(DEVICE))

# Load the trained face detection model
net = create_mb_tiny_fd(2)  # Assuming the model was trained for 2 classes (face and background)
net.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
net.to(DEVICE)
net.eval()

# Connect to SQLite database
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.landmarks_path)

# Function to extract facial encodings using dlib
def extract_face_encodings(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)
    if len(faces) == 0:
        return None
    shape = predictor(gray, faces[0])
    face_descriptor = np.array([shape.part(i).x for i in range(5)] +
                               [shape.part(i).y for i in range(5)], dtype=np.float64)
    return face_descriptor

# Function to compare face encodings
def compare_encodings(known_encoding, candidate_encoding, tolerance=0.6):
    distance = np.linalg.norm(known_encoding - candidate_encoding)
    return distance < tolerance

# Load known faces from the database
cursor.execute("SELECT name, encoding FROM faces")
known_faces = cursor.fetchall()

known_face_encodings = []
known_face_names = []
for name, encoding in known_faces:
    known_face_encodings.append(np.frombuffer(encoding, dtype=np.float64))
    known_face_names.append(name)

# Start video capture from the webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    logging.error("Error: Could not open webcam.")
    sys.exit()

logging.info("Starting real-time face recognition...")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = torch.from_numpy(rgb_frame).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            confidence, locations = net(tensor_frame)

        for i in range(confidence.shape[1]):
            if confidence[0, i, 1] > 0.5:  # Check confidence for face detection (index 1)
                bbox = locations[0, i].int().cpu().numpy()
                x1, y1, x2, y2 = bbox
                face_region = rgb_frame[y1:y2, x1:x2]

                face_encodings = extract_face_encodings(face_region)
                if face_encodings is not None:
                    matches = [compare_encodings(known_face_enc, face_encodings) for known_face_enc in known_face_encodings]
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = known_face_names[first_match_index]
                        print(f"Hello {name}")

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    conn.close()

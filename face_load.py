import cv2
import face_recognition
import sqlite3
import os


# Database connection
conn = sqlite3.connect('faces.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        encoding BLOB
    )
''')
conn.commit()
# Function to add a new face to the database
def add_new_face(name, image):
    """Adds a new face to the database.

    Args:
        name: The name of the person.
        image: The image of the person's face.
    """

    # Convert the image to RGB format (if necessary)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get face encoding from the image
    face_encoding = face_recognition.face_encodings(image)[0]

    # Store encoding and name in the database
    cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, face_encoding.tobytes()))
    conn.commit()
    print(f"Face added for {name}")

# Example Usage:
# Load image from file

# Directory containing face images
faces_directory = "faces"

# Get a list of all PNG files in the faces directory
png_files = [f for f in os.listdir(faces_directory) if f.endswith('.png')]

# Iterate over each PNG file and add it to the database
for png_file in png_files:
    # Extract the name from the filename (without the extension)
    name = os.path.splitext(png_file)[0]

    # Load the image
    image = cv2.imread(os.path.join(faces_directory, png_file))

    # Add the face to the database
    add_new_face(name, image)

# face=input("Enter the path of the image: ")
# # Get the name from the user
# name = input("Enter the name for the new face: ")
# image = cv2.imread(face) 
# # Add the face to the database
# add_new_face(name, image)

# Close the database connection
conn.close()
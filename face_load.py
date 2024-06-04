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
        name TEXT UNIQUE,
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
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) == 0:
        print(f"No face found in image for {name}")
        return

    face_encoding = face_encodings[0]

    # Check if the face already exists
    cursor.execute("SELECT id FROM faces WHERE name = ?", (name,))
    row = cursor.fetchone()

    if row:
        # Update existing face
        cursor.execute("UPDATE faces SET encoding = ? WHERE name = ?", (face_encoding.tobytes(), name))
        print(f"Face updated for {name}")
    else:
        # Insert new face
        cursor.execute("INSERT INTO faces (name, encoding) VALUES (?, ?)", (name, face_encoding.tobytes()))
        print(f"Face added for {name}")

    conn.commit()

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

# Close the database connection
conn.close()

import cv2
import os
import scipy.io
import numpy as np

def load_image(image_folder, image_filename):
    """
    Load an image from the specified folder.
    """
    image_path = os.path.join(image_folder, image_filename)
    image = cv2.imread(image_path)
    if image is None:
        print(f'Error: Failed to load image {image_filename}')
        return None
    else:
        print(f'Image loaded successfully: {image.shape}')
        return image



# Function to load metadata
def load_metadata(metadata_path):
    """
    Load metadata from the .mat file.
    """
    metadata = scipy.io.loadmat(metadata_path)
    return metadata


# Function to estimate face rectangle (here because it took long for it to parse correctly)
def estimate_face_rectangle(landmarks):
    """
    Estimate the face rectangle from facial landmarks.
    """
    if isinstance(landmarks, list) and len(landmarks) >= 2:
        x_coords = landmarks[::2]  # Even indices are x coordinates
        y_coords = landmarks[1::2]  # Odd indices are y coordinates
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        width = x_max - x_min
        height = y_max - y_min
        return [x_min, y_min, width, height]
    else:
        return [0, 0, 0, 0]  # Default value


# Function to fix illumination quality (had a problem with parsing too)
def fix_illumination_quality(value):
    """
    Fix the illumination quality value.
    """
    if value == 251:
        return 1  # Well-illuminated
    elif value == 252:
        return 2  # Poorly illuminated
    else:
        return 1  # Default to well-illuminated


# Function to parse a single metadata entry
def parse_metadata_entry(entry):
    """
    Parse a single metadata entry into a dictionary.
    """
    try:
        # Extract fields from the entry
        subject_id = entry[0][0][0][0]  # Subject ID
        image_id = entry[1][0][0][0]    # Image ID
        gender = entry[2][0]            # Gender
        age = entry[3][0][0]            # Age
        lighting = entry[4][0]          # Lighting
        frontal = entry[5][0]           # Frontal
        cropped = entry[6][0][0]        # Cropped
        estimated_points = entry[7][0][0]  # Estimated points
        year = entry[8][0][0]           # Year
        difficulty = entry[9][0]        # Difficulty
        emotion = entry[10][0][0]       # Emotion
        glasses_type = entry[11][0][0]  # Glasses type
        facial_landmarks = entry[12][0]  # Facial landmarks (17 points)
        face_rectangle = entry[13][0]   # Face rectangle [x, y, width, height]
        glasses_rectangle = entry[14][0]  # Glasses rectangle [x, y, width, height]
        illumination_quality = entry[15][0][0]  # Illumination quality
        filename = entry[16][0][0]      # Filename

        # Fix face_rectangle
        if isinstance(face_rectangle, np.ndarray) and face_rectangle.size == 4:
            face_rectangle = face_rectangle.tolist()  # Convert to list
        else:
            face_rectangle = estimate_face_rectangle(facial_landmarks)  # Estimate from landmarks

        # Fix glasses_rectangle
        if isinstance(glasses_rectangle, np.ndarray) and glasses_rectangle.size == 4:
            glasses_rectangle = glasses_rectangle.tolist()  # Convert to list
        else:
            glasses_rectangle = [0, 0, 0, 0]  # Default value

        # Fix illumination_quality
        illumination_quality = fix_illumination_quality(illumination_quality)

        # Fix filename
        if isinstance(filename, str):
            filename = filename
        else:
            filename = f"{subject_id}_{image_id}_{gender}_{age}_*"  # Default value

        # Return parsed metadata as a dictionary
        return {
            "subject_id": subject_id,
            "image_id": image_id,
            "gender": gender,
            "age": age,
            "lighting": lighting,
            "frontal": frontal,
            "cropped": cropped,
            "estimated_points": estimated_points,
            "year": year,
            "difficulty": difficulty,
            "emotion": emotion,
            "glasses_type": glasses_type,
            "facial_landmarks": facial_landmarks.tolist(),  # Convert to list
            "face_rectangle": face_rectangle,
            "glasses_rectangle": glasses_rectangle,
            "illumination_quality": illumination_quality,
            "filename": filename,
        }
    except (IndexError, KeyError, AttributeError) as e:
        print(f"Error parsing metadata entry: {e}")
        return None


# Function to parse the filename (comments are the outputs of the first image)
def parse_filename(filename):
    """
    Parse the filename into a dictionary of fields.
    """
    parts = filename.split("_")
    return {
        "subject_id": parts[0],  # AbdA
        "image_id": parts[1],    # 00001
        "gender": parts[2],      # m
        "age": int(parts[3]),    # 31
        "lighting": parts[4],    # i
        "frontal": parts[5],     # fr
        "cropped": parts[6],     # nc
        "estimated_points": parts[7],  # no
        "year": int(parts[8]),   # 2016
        "difficulty": int(parts[9]),  # 2
        "emotion": parts[10],    # e0
        "glasses_type": parts[11],  # Gn
        "illumination_quality": parts[12].split(".")[0],  # h (remove .jpg)
    }

# Function to match metadata to an image
def match_metadata_to_image(image_folder, filename, parsed_metadata):
    """
    Match metadata to an image using the filename.
    """
    parsed_filename = parse_filename(filename)
    subject_id = parsed_filename["subject_id"]
    image_id = parsed_filename["image_id"]

    # Find the corresponding metadata entry
    for entry in parsed_metadata:
        if entry["subject_id"] == subject_id and entry["image_id"] == image_id:
            return entry

    # If no match is found
    return None

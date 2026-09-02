import io
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
from PIL import Image
import cv2
import face_recognition

def load_image_to_rgb(image_input: Union[bytes, io.BytesIO, str, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Converts various image input formats into a standardized RGB numpy array (uint8).
    
    Supports:
        - Uploaded file-like objects (e.g. Streamlit UploadedFile or camera buffer)
        - Raw bytes
        - Filepath string
        - PIL Image object
        - Existing numpy array (BGR or RGB)
    """
    if isinstance(image_input, np.ndarray):
        # If OpenCV BGR image is passed (3 channels), convert to RGB
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            return image_input
        return image_input

    if isinstance(image_input, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_input))
        return np.array(image.convert("RGB"))

    if hasattr(image_input, "read"):  # File-like object (BytesIO, Streamlit UploadedFile)
        image_input.seek(0)
        image = Image.open(image_input)
        return np.array(image.convert("RGB"))

    if isinstance(image_input, str):  # File path
        return face_recognition.load_image_file(image_input)

    if isinstance(image_input, Image.Image):
        return np.array(image_input.convert("RGB"))

    raise TypeError(f"Unsupported image input type: {type(image_input)}")

def encode_face(image_input: Union[bytes, io.BytesIO, str, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Extracts the 128-dimensional facial feature vector from an enrollment image.
    
    Enforces strict validation:
        - Exactly ONE face must be present in the photo.
        - Raises ValueError with user-friendly explanations if 0 or >1 faces are found.
        
    Args:
        image_input: The input image.
        
    Returns:
        128-d numpy array representing facial embeddings.
    """
    rgb_image = load_image_to_rgb(image_input)
    
    # Locate faces using HOG-based detector (fast and accurate for front-facing photos)
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    
    num_faces = len(face_locations)
    if num_faces == 0:
        raise ValueError("No face detected in the photo. Please ensure the face is clearly visible, upright, and well-lit.")
    
    if num_faces > 1:
        raise ValueError(f"Multiple faces detected ({num_faces}). Enrollment requires an individual photo with exactly one person.")
    
    # Compute 128-dimensional facial embeddings
    encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)
    
    if not encodings or len(encodings) == 0:
        raise ValueError("Failed to extract facial features from the detected face. Please try a clearer photograph.")
        
    return encodings[0]

def draw_annotations(rgb_image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draws visually clear bounding boxes and label badges around detected faces.
    
    Args:
        rgb_image: Base RGB image.
        detections: List of detection result dictionaries.
        
    Returns:
        Annotated RGB image.
    """
    annotated = rgb_image.copy()
    
    for det in detections:
        top, right, bottom, left = det["location"]
        is_match = det["is_match"]
        name = det["name"]
        dist = det["distance"]
        
        # Color scheme: Emerald Green for recognized, Coral Red for unknown
        box_color = (46, 204, 113) if is_match else (231, 76, 60)
        
        # Draw bounding rectangle
        cv2.rectangle(annotated, (left, top), (right, bottom), box_color, 2)
        
        # Format label text
        if is_match:
            label = f"{name} ({dist:.2f})"
        else:
            label = "Unknown"
            
        # Draw label background badge above or below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position badge
        label_top = max(top - text_height - 10, 0)
        label_bottom = max(top, text_height + 10)
        
        cv2.rectangle(
            annotated, 
            (left, label_top), 
            (left + text_width + 12, label_bottom), 
            box_color, 
            cv2.FILLED
        )
        cv2.putText(
            annotated, 
            label, 
            (left + 6, label_bottom - 5), 
            font, 
            font_scale, 
            (255, 255, 255), 
            thickness, 
            cv2.LINE_AA
        )
        
    return annotated

def recognize_faces(
    image_input: Union[bytes, io.BytesIO, str, np.ndarray],
    known_workers: List[Dict],
    tolerance: float = 0.50
) -> Tuple[List[Dict], np.ndarray]:
    """
    Detects and matches all faces in a frame against a list of enrolled workers.
    
    Distance metric:
        - Euclidean distance between 128-d vectors.
        - Distance < tolerance (default 0.50) indicates a confident match. Lower is stricter.
        
    Args:
        image_input: Camera frame or uploaded image.
        known_workers: List of worker dicts [{'id': int, 'name': str, 'encoding': np.ndarray}].
        tolerance: Distance threshold for positive face verification (0.50 recommended).
        
    Returns:
        Tuple of:
            - List of detection dicts: [{
                'worker_id': int or None,
                'name': str,
                'distance': float,
                'location': (top, right, bottom, left),
                'is_match': bool
              }]
            - Annotated RGB image with bounding boxes and labels.
    """
    rgb_image = load_image_to_rgb(image_input)
    
    # Locate all faces in the incoming frame
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    
    if not face_locations:
        return [], rgb_image
        
    face_encodings = face_recognition.face_encodings(rgb_image, known_face_locations=face_locations)
    
    detections: List[Dict] = []
    
    if not known_workers:
        # If no enrolled workers exist, mark all detected faces as Unknown
        for loc in face_locations:
            detections.append({
                "worker_id": None,
                "name": "Unknown",
                "distance": 1.0,
                "location": loc,
                "is_match": False
            })
        return detections, draw_annotations(rgb_image, detections)
    
    # Prepare known encodings array
    known_encodings = [w["encoding"] for w in known_workers]
    
    for face_loc, face_encoding in zip(face_locations, face_encodings):
        # Compute Euclidean distance to every registered worker
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        
        if best_distance <= tolerance:
            matched_worker = known_workers[best_idx]
            detections.append({
                "worker_id": matched_worker["id"],
                "name": matched_worker["name"],
                "distance": best_distance,
                "location": face_loc,
                "is_match": True
            })
        else:
            detections.append({
                "worker_id": None,
                "name": "Unknown",
                "distance": best_distance,
                "location": face_loc,
                "is_match": False
            })
            
    annotated_image = draw_annotations(rgb_image, detections)
    return detections, annotated_image

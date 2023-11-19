from PIL import Image
from pathlib import Path
from autocrop import Cropper
import json
        
def detect_face_and_resize(image_path):
    """Crop around face and resize an image to 512x512."""
    
    cropper = Cropper(width=512, height=512)
    cropped_array = cropper.crop(image_path)
    
    if cropped_array is None:
        raise ValueError("No face detected in image.")
    
    return Image.fromarray(cropped_array)


def resize_and_center_crop(img_path):
    img = Image.open(img_path)
    width, height = img.size

    # Calculate dimensions for the crop
    if width > height:
        left = (width - height) / 2
        top = 0
        right = (width + height) / 2
        bottom = height
    else:
        top = (height - width) / 2
        left = 0
        bottom = (height + width) / 2
        right = width

    # Crop to a square
    img = img.crop((left, top, right, bottom))

    # Resize to 512x512
    img = img.resize((512, 512))

    return img
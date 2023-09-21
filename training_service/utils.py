from PIL import Image
from pathlib import Path
import tarfile
import random
from itertools import chain
from autocrop import Cropper
import json


def extract_tar(tar_obj, extract_path):
    
    with tarfile.open(fileobj=tar_obj) as tar:
        tar.extractall(extract_path)
        
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



def preprocess_images(tar_obj):
    
    tmp_path = Path(f"/tmp/{random.randint(0, 1000000)}")
    while tmp_path.exists():
        tmp_path = Path(f"/tmp/{random.randint(0, 1000000)}")
    tmp_path.mkdir(parents=True, exist_ok=False)
    
    extract_tar(tar_obj, tmp_path)

    # check if config.json exists in the train path
    if (tmp_path / "config.json").exists():
        tuning_config = json.load((tmp_path / "config.json").open())
    else:
        tuning_config = {}  

    face_preprocessing = tuning_config.pop("face_preprocessing", False)
    
    dest_path = tmp_path / "cropped"
    dest_path.mkdir(parents=True, exist_ok=True)

    crop_func = detect_face_and_resize if face_preprocessing else resize_and_center_crop
    
    for n,img_path in enumerate(chain(tmp_path.glob("*.[jJ][pP]*[Gg]"),tmp_path.glob("*.[Pp][Nn][Gg]"))):
        try:
            cropped = crop_func(img_path.as_posix())
            cropped.save(dest_path / f"image_{n}.png")
        except ValueError:
            print(f"Could not detect face in {img_path}. Skipping.")
            continue
    
    
    return dest_path, tuning_config
from PIL import Image
from pathlib import Path
import tarfile
import random
from itertools import chain
from autocrop import Cropper


def extract_tar(tar_obj, extract_path):
    
    with tarfile.open(fileobj=tar_obj) as tar:
        tar.extractall(extract_path)
        
def detect_and_resize(image_path):
    """Crop around face and resize an image to 512x512."""
    
    cropper = Cropper(width=512, height=512)
    cropped_array = cropper.crop(image_path)
    
    if cropped_array is None:
        raise ValueError("No face detected in image.")
    
    return Image.fromarray(cropped_array)

def preprocess_images(tar_obj):
    
    tmp_path = Path(f"/tmp/{random.randint(0, 1000000)}")
    while tmp_path.exists():
        tmp_path = Path(f"/tmp/{random.randint(0, 1000000)}")
    tmp_path.mkdir(parents=True, exist_ok=False)
    
    extract_tar(tar_obj, tmp_path)
    
    dest_path = tmp_path / "cropped"
    dest_path.mkdir(parents=True, exist_ok=True)
    
    for n,img_path in enumerate(chain(tmp_path.glob("*.[jJ][pP][Gg]"),tmp_path.glob("*.[Pp][Nn][Gg]"))):
        try:
            cropped = detect_and_resize(img_path.as_posix())
            cropped.save(dest_path / f"image_{n}.png")
        except ValueError:
            print(f"Could not detect face in {img_path}. Skipping.")
            continue
    
    return dest_path

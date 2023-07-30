from PIL import Image
from pathlib import Path
import tarfile
import random
from itertools import chain
from autocrop import Cropper


def extract_tar(tar_obj, extract_path):
    
    with tarfile.open(fileobj=tar_obj) as tar:
        tar.extractall(extract_path)

def resize(image_path):
    target_size = (512, 512)
    image = Image.open(image_path)
    
    # Resize the image while maintaining aspect ratio
    image.thumbnail(target_size, Image.ANTIALIAS)
    image_size = image.size
    
    x = (image_size[0] - target_size[0]) // 2
    y = (image_size[1] - target_size[1]) // 2
    image = image.crop((x, y, x + target_size[0], y + target_size[1]))
    
    return image
    
def detect_and_resize(image_path):
    """Crop around face and resize an image to 512x512."""
    
    cropper = Cropper(width=512, height=512)
    cropped_array = cropper.crop(image_path)
    
    if cropped_array is None:
        raise ValueError("No face detected in image.")
    
    return Image.fromarray(cropped_array)


def preprocess_training_images(image_path, face_cropping):
    
    input_path = Path(image_path)
    
    dest_path = input_path / "cropped"
    dest_path.mkdir(parents=True, exist_ok=True)
    
    for n,img_path in enumerate(chain(input_path.glob("*.[jJ][pP][Gg]"),input_path.glob("*.[Pp][Nn][Gg]"))):
        try:
            if face_cropping:
                cropped = detect_and_resize(img_path.as_posix())
            else:
                cropped = resize(img_path.as_posix())
                
            cropped.save(dest_path / f"image_{n}.png")
        except ValueError:
            print(f"Could not detect face or crop image in {img_path}. Skipping.")
            continue
    
    return dest_path
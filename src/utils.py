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
    # Calculate the aspect ratio
    width, height = image.size
    aspect_ratio = width / height
    
    # Calculate the new dimensions for scaling down
    if width > height:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    
    # Resize the image while maintaining aspect ratio
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
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

def preprocess_training_images(image_path):#, face_cropping):
    
    input_path = Path(image_path)
    
    dest_path = input_path / "cropped"
    dest_path.mkdir(parents=True, exist_ok=True)
    
    for n,img_path in enumerate(chain(input_path.glob("*.[jJ][pP][Gg]"),input_path.glob("*.[Pp][Nn][Gg]"))):
        try:
            # if face_cropping:
            cropped = detect_and_resize(img_path.as_posix())
            # else:
            #     cropped = resize(img_path.as_posix())
                
            cropped.save(dest_path / f"image_{n}.png")
        except ValueError:
            print(f"Could not detect face or crop image in {img_path}. Skipping.")
            continue
    
    return dest_path
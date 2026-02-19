import io
from PIL import Image
from rembg import remove

def isolate_leaf(image_bytes):
    print("Segmenting image and removing background...")
    # 1. Eliminate the background using rembg
    subject_only = remove(image_bytes)
    # 2. Convert the isolated subject back into a standard RGB image
    segmented_img = Image.open(io.BytesIO(subject_only)).convert("RGB")
    
    return segmented_img
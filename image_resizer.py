import os
from PIL import Image

INPUT_DIR = "train_images"
OUTPUT_DIR = "train_ready"
TARGET_RES = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

def center_crop(img):
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    right = left + s
    bottom = top + s
    return img.crop((left, top, right, bottom))

for folder in os.listdir(INPUT_DIR):
    folder_path = os.path.join(INPUT_DIR, folder)
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        img = center_crop(img)
        img = img.resize((TARGET_RES, TARGET_RES), Image.LANCZOS)

        outname = os.path.splitext(fname)[0] + ".png"  # StyleGAN prefers PNG
        new_path = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(new_path, exist_ok=True)
        img.save(os.path.join(OUTPUT_DIR, folder, outname))

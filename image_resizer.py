import json, os
from PIL import Image

INPUT_DIR = "train_images"
OUTPUT_DIR = "train_ready"
TARGET_RES = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)
name_dict = {}
class_dict = {
    "Normal": 1,
    "Cataract": 2,
    "Conjunctivitis": 3,
    "Uveitis": 4,
    "WarpedEyelid": 5,
}

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

        outname = os.path.splitext(fname)[0] + "_" + folder + ".png"  # StyleGAN prefers PNG
        img.save(os.path.join(OUTPUT_DIR, outname))
        name_dict[outname] = class_dict[folder]

dataset_json = {
    "labels": name_dict,
    "label_dim": 5
}
with open(os.path.join(OUTPUT_DIR, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=2)
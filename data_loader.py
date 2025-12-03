import pandas as pd
import io
import os
from PIL import Image

PARQUET_PATH = "train-00000-of-00001-dc2528734ece7546.parquet"
OUTPUT_FOLDER = "train_images"

# Make output folder if missing
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load parquet
df = pd.read_parquet(PARQUET_PATH)

# Replace these with the actual column names:
IMAGE_COL = "image"
LABEL_COL = "label"

for idx, row in df.iterrows():

    # Extract raw image bytes (HuggingFace parquet uses this structure)
    img_bytes = row[IMAGE_COL]["bytes"]

    # Decode to PIL image
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Build filename: label_index.png
    label = row[LABEL_COL]
    filename = f"{label}_{idx}.png"

    save_path = os.path.join(OUTPUT_FOLDER, filename)
    img.save(save_path)

print("Done! Images saved to:", OUTPUT_FOLDER)

import gdown
import os

# Folder & nama file tujuan
MODEL_PATH = "model/model_sampah.h5"
DRIVE_ID = "13lwGtCrJpWZmwXQNHAbc9qu6JPxIAodd"  # ← Ganti dengan ID kamu

# Buat folder jika belum ada
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Download jika belum ada
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_ID}", MODEL_PATH, quiet=False)
else:
    print("Model already exists.")

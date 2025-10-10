import os
import requests
import zipfile
import io

DATA_DIR = os.path.join("data", "pill_dataset")
os.makedirs(DATA_DIR, exist_ok=True)

# ✅ Public sample dataset hosted on GitHub (no login needed)
DATA_URL = "https://github.com/dataprofessor/data/raw/master/pill-image-sample.zip"

print("📥 Downloading pill dataset (sample)...")
response = requests.get(DATA_URL)
if response.status_code == 200:
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(DATA_DIR)
    print(f"✅ Dataset downloaded and extracted to: {DATA_DIR}")
else:
    print(f"❌ Download failed with status code: {response.status_code}")

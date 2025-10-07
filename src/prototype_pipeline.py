import sys
import cv2
import easyocr
from pathlib import Path


def identify_text_from_image(image_path):
    # Initialize OCR reader
    reader = easyocr.Reader(['en'])

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print("❌ Could not read image:", image_path)
        return

    # Run OCR
    results = reader.readtext(img)

    print(f"🧾 OCR Results for {image_path.name}:")
    for (bbox, text, prob) in results:
        print(f"→ {text} (Confidence: {prob:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/prototype_pipeline.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    identify_text_from_image(image_path)

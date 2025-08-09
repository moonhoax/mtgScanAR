import os
import requests
import subprocess

# --- Configuration ---
SCRYFALL_IMAGE_URL = "https://cards.scryfall.io/normal/front/7/2/721e1b17-888f-462b-9f95-7887fd0b18c6.jpg"  # Example: The One Ring
CARD_NAME = "The One Ring"
OUTPUT_DIR = "/tmp/public/assets/mindar"  # Where .mind files will go
TEMP_IMAGE_PATH = "/tmp/temp_card.jpg"

def sanitize_filename(name):
    return name.lower().replace(" ", "_").replace(",", "").replace("'", "")

def download_image(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"[✓] Downloaded image to {save_path}")
    else:
        raise Exception(f"Failed to download image: {response.status_code}")

def generate_mind_marker(image_path, output_path):
    result = subprocess.run([
        "mindar", "image", "-i", image_path, "-o", output_path
    ], capture_output=True, text=True)

    if result.returncode == 0:
        print(f"[✓] Marker generated at {output_path}.mind")
    else:
        print("[!] Error generating marker:")
        print(result.stderr)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = sanitize_filename(CARD_NAME)
    mind_output_path = os.path.join(OUTPUT_DIR, file_name)

    # Step 1: Download Scryfall image
    download_image(SCRYFALL_IMAGE_URL, TEMP_IMAGE_PATH)

    # Step 2: Generate .mind marker
    generate_mind_marker(TEMP_IMAGE_PATH, mind_output_path)

    # Step 3: Clean up
    if os.path.exists(TEMP_IMAGE_PATH):
        os.remove(TEMP_IMAGE_PATH)

if __name__ == "__main__":
    main()

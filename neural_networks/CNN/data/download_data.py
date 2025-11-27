import gdown
import os
import zipfile

def download_and_extract(url, output_path="data/dataset.zip", extract_to="data/"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print("Downloading dataset...")
    gdown.download(url, output_path, quiet=False)
    print("Download complete — extracting…")
    with zipfile.ZipFile(output_path, "r") as z:
        z.extractall(extract_to)
    print("Extraction complete. Data available in:", extract_to)

if __name__ == "__main__":
    # Replace this URL with your shared Google Drive URL
    url = "https://drive.google.com/uc?id=1Xodq1tBD-udPhHA7x0sFXqPap6zyDVSz"
    download_and_extract(url)

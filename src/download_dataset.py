import os
import urllib.request
import zipfile
from tqdm import tqdm
from PIL import Image

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def verify_and_clean_images(data_dir):
    print("Verifying images and removing corrupted ones...")
    removed = 0
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # verify that it is, in fact an image
            except Exception as e:
                print(f"Bad image found: {file_path}. Deleting...")
                try:
                    os.remove(file_path)
                    removed += 1
                except FileNotFoundError:
                    pass
                except PermissionError:
                    print(f"Permission denied to delete {file_path}")
    print(f"Cleaning complete. Removed {removed} corrupted images.")

def main():
    url = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"
    
    # Ensure data directory exists
    os.makedirs("../data", exist_ok=True)
    zip_path = "../data/kagglecatsanddogs_5340.zip"
    extract_path = "../data/"

    # Step 1: Download
    if not os.path.exists(zip_path) and not os.path.exists(os.path.join(extract_path, "PetImages")):
        print("Downloading Cats and Dogs dataset...")
        download_url(url, zip_path)
    else:
        print("Dataset zip already exists or already extracted.")

    # Step 2: Extract
    if not os.path.exists(os.path.join(extract_path, "PetImages")):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")

    # Step 3: Clean
    verify_and_clean_images(os.path.join(extract_path, "PetImages"))

if __name__ == "__main__":
    main()

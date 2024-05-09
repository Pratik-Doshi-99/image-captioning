import urllib.request
import zipfile
import os

# Replace with your ZIP file URL
#DATASET = 'https://gitlab.nrp-nautilus.io/pratikdoshi/data_files/-/raw/main/image-captioning-project/flickr8k.zip' #Flick8k
DATASET = 'https://gitlab.nrp-nautilus.io/pratikdoshi/data_files/-/raw/main/image-captioning-project/flickr8k_m.zip'  #Test dataset
# Specify the directory where the file should be downloaded
DOWNLOAD_DIRECTORY = os.path.join('.','data') # current directory
# the name of the directory inside the zip which has the Images sub directory and captions.txt file
DEST_DIRECTORY = 'flickr8k_m'

def download_and_unzip(url, download_directory, dest_dir, filename='data.zip'):

    target_dir = os.path.join(download_directory, dest_dir)
    if os.path.isdir(target_dir):
        print(f'Target dir={target_dir} exists. Skipping Download')
        return target_dir


    # Create the download directory if it does not exist
    os.makedirs(download_directory, exist_ok=True)
    
    # Path to save the downloaded ZIP file
    zip_file_path = os.path.join(download_directory, filename)
    
    # Download the ZIP file
    print(f"Downloading ZIP file from {url}...")
    urllib.request.urlretrieve(url, zip_file_path)
    print(f"ZIP file downloaded to {zip_file_path}")
    
    # Unzip the contents
    print(f"Unzipping the contents of {zip_file_path}...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_directory)
        print(f"Contents extracted to {download_directory}")
    
    # Optionally, remove the ZIP file after extraction
    os.remove(zip_file_path)
    print(f"ZIP file removed: {zip_file_path}")

    return target_dir


# Download and unzip the ZIP file
print(download_and_unzip(DATASET, DOWNLOAD_DIRECTORY, DEST_DIRECTORY))

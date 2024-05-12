import urllib.request
import zipfile
import os
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Replace with your ZIP file URL
#DATASET = 'https://gitlab.nrp-nautilus.io/pratikdoshi/data_files/-/raw/main/image-captioning-project/flickr8k.zip' #Flick8k
DATASET = 'https://gitlab.nrp-nautilus.io/pratikdoshi/data_files/-/raw/main/image-captioning-project/flickr8k_m.zip'  #Test dataset
# Specify the directory where the file should be downloaded
DOWNLOAD_DIRECTORY = os.path.join('.','data') # current directory
# the name of the directory inside the zip which has the Images sub directory and captions.txt file
DEST_DIRECTORY = 'flickr8k'

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


def get_default_device():
  """Pick GPU if available, else CPU"""
  if torch.cuda.is_available():
      return torch.device('cuda')
  else:
      return torch.device('cpu')

device = get_default_device()


def get_flickr8k_captions(captions_path=None, skip_rows=1):
    if captions_path is None:
        captions_path = os.path.join(DOWNLOAD_DIRECTORY, DEST_DIRECTORY,'captions.txt')
    captions = []
    with open(captions_path, 'r') as caption_file:
        captions = caption_file.readlines()
    
    captions = [c.strip('\n').split(',') for c in captions[skip_rows:]]
    return captions


    

class ImageDataset(Dataset):
    def __init__(self, img_captions, img_size=(224,224), root=None):
        super().__init__()
        self.device = get_default_device()
        self.root = os.path.join(DOWNLOAD_DIRECTORY, DEST_DIRECTORY, 'Images') if root is None else root
        
        #initializing the transformation set with the default conversion
        self.img_captions = img_captions # [(img_name, caption),(img_name, caption)]
        self.default_transformation = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
)

    def __len__(self):
        return len(self.img_captions)

    def __getitem__(self, index):
        #print(index)
        img_path = os.path.join(self.root, self.img_captions[index][0])
        caption = self.img_captions[index][1]
        image = Image.open(img_path).convert('RGB')
        return self.default_transformation(image), caption
      




# Download and unzip the ZIP file
print(download_and_unzip(DATASET, DOWNLOAD_DIRECTORY, DEST_DIRECTORY))

if __name__ =='__main__':
    img_captions = get_flickr8k_captions()
    print(img_captions[:2])
    dataset = ImageDataset(img_captions)
    for i in torch.randint(0, len(dataset), (10,1)):
        print(dataset[i][0].shape, dataset[i][1])

import pandas as pd
import Vocabulary as v 
import os
from PIL import Image 
import torch
from torch.utils.data import Dataset, DataLoader
import captionPreprocessing

class FlickrDataset(Dataset):

    def __init__(self,root_dir,captions_file,transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        
        #Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        #Preprocess the captions
        self.vocab = v.Vocabulary(vocab_file='vocab.pkl',vocab_from_file=True)
        self.caption_preprocessor = captionPreprocessing.CaptionProprocessor()
        self.max_caption_length = self.caption_preprocessor.max_Caption_Length(self.captions)

        
        #Initialize vocabulary and build vocab
        self.vocab = v.Vocabulary(vocab_file='vocab.pkl',vocab_from_file= True)
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        caption = self.captions[index]
        img_name = self.imgs[index]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        #apply the transformation to the image
        if self.transform is not None:
            img = self.transform(img)
        
        # Convert caption to indices
        caption_vec = self.caption_preprocessor.convertCaptionToIndices(caption,self.max_caption_length,self.vocab)
        
        return img, torch.tensor(caption_vec)
    

def get_loader(root_folder,captions_file,transform,batch_size=32,num_workers=2,shuffle=True):
    dataset = FlickrDataset(root_folder,captions_file,transform)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return dataset , data_loader
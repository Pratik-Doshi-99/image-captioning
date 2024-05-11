import torch
import torch.nn as nn
import torchvision
import data
from torchinfo import summary
import pickle
import os

class VITEncoder(nn.Module):
    def __init__(self):
        super(VITEncoder, self).__init__()
        self.vit = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1 )
        self.vit.heads.head = nn.Identity()
        # num_features = self.vit.heads.head.in_features
        # self.vit.heads.head = nn.Sequential(
        #             nn.Linear(num_features, output_dimensions),
        #             #nn.ReLU(inplace=True),
        #             #nn.Dropout(0.5),
        #             #nn.Linear(linear_dim
        # , output_dimensions),
        #         )

    def forward(self, x):
        return self.vit(x)


def save_state(embeddings, file_name):
    dest = os.path.join('.','embeddings')
    os.makedirs(dest, exist_ok=True)
    
    with open(os.path.join(dest,file_name), 'wb') as file:
        # Serialize the object using pickle and write it to the file
        pickle.dump(embeddings, file)
        print(f"Object saved to '{file_name}'")

def compute_img_embeddings(model, batch_size = 8):
    img_captions = data.get_flickr8k_captions()
    dataset = data.ImageDataset(img_captions)
    device = data.get_default_device()
    visited = set()
    model.to(device)
    embeddings = {}
    batch = []
    save_at = batch_size * 10
    file_index = 1
    for i, c in enumerate(img_captions):
        if c[0] in visited:
            continue
        img, caption = dataset[i]
        print(i,c[0],caption,sep=',')
        visited.add(c[0])
        batch.append((img, c[0]))
        if len(batch) == batch_size:
            output = model(torch.stack([b[0] for b in batch]).to(device))
            updated_dict = {b[1]:output[i] for i,b in enumerate(batch)}
            embeddings.update(updated_dict)
            batch = []
        if i > 0 and i % save_at == 0:
            save_state(embeddings, f'embeddings_{file_index}.bin')
            file_index += 1
            embeddings = {}
    
    save_state(embeddings, f'embeddings_{file_index}.bin')

            

        

def compute_embeddings_fast(model, batch_size = 8):
    img_captions = data.get_flickr8k_captions()
    dataset = data.ImageDataset(img_captions)
    device = data.get_default_device()
    image_index = {}
    model.to(device)
    cumm_tensor = None
    batch = []
    for i, c in enumerate(img_captions):
        if c[0] in image_index:
            continue
        img, caption = dataset[i]
        #img = img.to(device)
        print(i,c[0],caption,sep=',')
        image_index[c[0]] = i
        batch.append(img)
        if len(batch) == batch_size:
            batch = torch.stack(batch).to(device)
            output = model(batch)
            cumm_tensor = torch.cat((cumm_tensor, output),0) if cumm_tensor is not None else output
            output = None
            print('Tensor Shape',cumm_tensor.shape)
            batch = []

    if batch:
        batch = torch.stack(batch)
        output = model(batch)
        cumm_tensor = torch.cat((cumm_tensor, output),0) if cumm_tensor is not None else output
        batch = []
    save_state(cumm_tensor, 'embeddings.bin')
    save_state(image_index, 'img_embedding_map.bin')




if __name__ == '__main__':
    model = VITEncoder()
    #print(summary(model, input_size=(1,3,224,224),col_names=["input_size", "output_size", "num_params", "trainable"])) 
    #compute_img_embeddings(model)
    compute_embeddings_fast(model, batch_size = 8)
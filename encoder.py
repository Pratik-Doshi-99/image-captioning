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

def compute_img_embeddings(model, batch_size = 4):
    img_captions = data.get_flickr8k_captions()
    dataset = data.ImageDataset(img_captions)
    device = data.get_default_device()
    model.to(device)
    embeddings = {}
    batch = []
    save_at = batch_size * 10
    file_index = 1
    for i, c in enumerate(img_captions):
        if c[0] in embeddings:
            continue
        img, caption = dataset[i]
        print(i,c[0],caption,sep=',')
        embeddings[c[0]] = None
        batch.append((img, c[0]))
        if len(batch) == batch_size:
            output = model(torch.stack([b[0] for b in batch]))
            updated_dict = {b[1]:output[i] for i,b in enumerate(batch)}
            embeddings.update(updated_dict)
            batch = []
        if i > 0 and i % save_at == 0:
            save_state(embeddings, f'embeddings_{file_index}.bin')
            file_index += 1
            for k in embeddings:
                embeddings[k] = None
    
    save_state(embeddings, f'embeddings_{file_index}.bin')

            

        




if __name__ == '__main__':
    model = VITEncoder()
    #print(summary(model, input_size=(1,3,224,224),col_names=["input_size", "output_size", "num_params", "trainable"]))
    compute_img_embeddings(model)
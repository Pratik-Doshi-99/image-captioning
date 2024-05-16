import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        if(train_CNN == False):
            for param in resnet.parameters():
                param.requires_grad_(False)
        resnet.fc = nn.Linear(resnet.fc.in_features, embed_size)


    def forward(self, images):
        features = self.resnet(images)
        return features
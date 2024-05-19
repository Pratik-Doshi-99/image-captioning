import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V2)
        if(train_CNN == False):
            for name, param in self.resnet.named_parameters():
                if "fc.weight" in name or "fc.bias" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = train_CNN
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)


    def forward(self, images):
        features = self.resnet(images)
        return features
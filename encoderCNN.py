import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.resnet = models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, embed_size)
        self.resnet.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        if(train_CNN == False):
            for name, param in self.resnet.named_parameters():
                if "fc" in name or "bn" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = train_CNN
        else: 
            for name, param in self.resnet.named_parameters():
                param.requires_grad = True


    def forward(self, images):
        features = self.resnet(images)
        return features
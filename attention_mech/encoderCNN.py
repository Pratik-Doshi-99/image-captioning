import torch.nn as nn
import torchvision.models as models


class EncoderCNNAttention(nn.Module):
    def __init__(self, train = False):
        super(EncoderCNNAttention, self).__init__()
        
        self.resnet = models.resnet50(weights= models.ResNet50_Weights.IMAGENET1K_V2)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        for n,p in self.resnet.named_parameters():
            p.requires_grad = train

        self.dim = 2048

    def forward(self, x):
        x = self.resnet(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1, x.size(-1))
        return x


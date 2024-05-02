import torch
import torch.nn as nn
import torchvision


class VITEncoder(nn.Module):
    def __init__(self, output_dimensions):
        super(VITEncoder, self).__init__()
        self.vit = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1)
        num_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Sequential(
                    nn.Linear(num_features, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(1024, output_dimensions),
                )

    def forward(self, x):
        return self.vit(x)



if __name__ == '__main__':
    model = VITEncoder(12)
    sample_tensor = torch.rand(1, 3,224,224)
    print('Starting forward pass')
    x = model(sample_tensor)
    print(x)
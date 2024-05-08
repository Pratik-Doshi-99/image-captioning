import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from encoder import VITEncoder

class BaselineGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super(BaselineGRU, self).__init__()
        
        # Load pretrained ViT model from timm library
        self.vit = VITEncoder(output_dimensions=embed_size)
        
        # GRU for generating captions
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer to generate predictions for each word in the vocabulary
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, image, captions, hidden=None):
        # Process the image through the ViT model
        image_features = self.vit(image)  # shape: (batch_size, embed_size)
        
        # Pass the captions and image features through the GRU
        # image_features is unsqueezed to match the expected dimensions (batch_size, 1, embed_size)
        output, hidden = self.gru(captions, image_features.unsqueeze(0))
        
        # Pass the output through the fully connected layer to get the vocabulary size
        output = self.fc(output)
        
        return output, hidden


if __name__ == '__main__':
    # Hyperparameters
    vocab_size = 10000  # Size of the vocabulary (should match the dataset)
    embed_size = 256    # Embedding size
    hidden_size = 512   # Hidden size in GRU

    # Instantiate the model
    model = BaselineGRU(vocab_size, embed_size, hidden_size)

    # You would then define loss and optimizer, and use a dataset of images and captions to train the model
    # Example loss function and optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

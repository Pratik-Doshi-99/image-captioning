import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class decoderLSTM(nn.Module):
    def __init__(self,embed_size, hidden_size, vocab_size, num_layers=1,dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers=num_layers,batch_first=True)
        self.fcn = nn.Linear(hidden_size,vocab_size)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, features, captions):
        
        #vectorize the caption
        embeds = self.embedding(captions) # TODO : can also remove <end> / <pad> token 
        
        #concat the features and captions
        x = torch.cat((features.unsqueeze(1),embeds),dim=1) 
        x,_ = self.lstm(x)
        x = self.fcn(x)
        return x
    

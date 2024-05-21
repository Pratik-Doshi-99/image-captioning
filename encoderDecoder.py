import torch
import torch.nn as nn
import decoderLSTM
import encoderCNN 
class EncoderDecoder(nn.Module):
    def __init__(self,embed_size, hidden_size, vocab_size, num_layers=3,drop_prob=0.3,train_CNN=False):
        super().__init__()
        self.encoder = encoderCNN.EncoderCNN(embed_size, train_CNN= train_CNN)
        self.decoder = decoderLSTM.decoderLSTM(embed_size,hidden_size,vocab_size,num_layers,drop_prob)
    
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
    
    def get_caption_for_image(self, image, vocabulary, max_length=20):
        result_caption = []
        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0) # unsqueeze to add batch dimension
            states = None
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0)) # squeeze to remove batch dimension
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoder.embedding(predicted).unsqueeze(0)
                if vocabulary.get_word(predicted.item()) == "<end>":
                    break
        return [vocabulary.get_word(idx) for idx in result_caption]
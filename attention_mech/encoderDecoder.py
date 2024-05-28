import torch
import torch.nn as nn
import decoderLSTM
import encoderCNN 


class EncoderDecoderAttention(nn.Module):
    def __init__(self,embed_size, hidden_size, vocab_size):
        super().__init__()
        self.encoder = encoderCNN.EncoderCNNAttention()
        self.decoder = decoderLSTM.decoderAttentionLSTM(vocab_size, self.encoder.dim, embed_size, hidden_size)
    
    def forward(self, images, captions):
        img_features = self.encoder(images)
        preds, alpha = self.decoder(img_features, captions)
        return preds, alpha
    
    # def get_caption_for_image(self, image, vocabulary, max_length=20):
    #     result_caption = []
    #     with torch.no_grad():
    #         x = self.encoder(image).unsqueeze(0) # unsqueeze to add batch dimension
    #         states = None
    #         for _ in range(max_length):
    #             hiddens, states = self.decoder.lstm(x, states)
    #             output = self.decoder.linear(hiddens.squeeze(0)) # squeeze to remove batch dimension
    #             predicted = output.argmax(1)
    #             result_caption.append(predicted.item())
    #             x = self.decoder.embedding(predicted).unsqueeze(0)
    #             if vocabulary.get_word(predicted.item()) == "<end>":
    #                 break
    #     return [vocabulary.get_word(idx) for idx in result_caption]

    def get_caption_for_image(self, image, vocabulary):
        with torch.no_grad():
            img_features = self.encoder(image.unsqueeze(0))
            caption = self.decoder.caption(img_features, 17)
        return [vocabulary.get_word(idx) for idx in caption]

    
    
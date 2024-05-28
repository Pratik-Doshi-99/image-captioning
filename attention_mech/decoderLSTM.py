from attention import Attention
import torch.nn as nn
import torch
from dataset import device

class decoderAttentionLSTM(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim, embed_size, hidden_size):
        super(decoderAttentionLSTM, self).__init__()
        #Teaching Forcing set to true by default to make results comparable to baseline
        self.use_tf = True
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(hidden_size, vocabulary_size)
        self.dropout = nn.Dropout()

        self.attention = Attention(encoder_dim, hidden_size)
        self.embedding = nn.Embedding(vocabulary_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)

    def forward(self, img_features, captions):
        """
        We can use teacher forcing during training. For reference, refer to
        https://www.deeplearningbook.org/contents/rnn.html

        """
        # image features: batch x 49 x 2048
        # captions: batch number of list of indices of words

        batch_size = img_features.size(0)

        h, c = self.get_init_lstm_state(img_features)
        max_timespan = max([len(caption) for caption in captions]) - 1

        # prev_words = torch.zeros(batch_size, 1).long().cuda()
        prev_words = torch.zeros(batch_size, 1).long().to(device)
        if self.use_tf:
            embedding = self.embedding(captions) if self.training else self.embedding(prev_words)
        else:
            embedding = self.embedding(prev_words)

        # preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).cuda()
        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).to(device)
        # preds = batch * caption length * vocab size 
        # alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).cuda()
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).to(device)
        # alphas = batch * caption length * 49

        for t in range(max_timespan):
            context, alpha = self.attention(img_features, h)
            # context = batch * encoder_dim
            gate = self.sigmoid(self.f_beta(h))
            # gate = batch * encoder_dim (f_beta maps hidden_dim to encoder_dim)
            gated_context = gate * context
            # gated_context = batch * encoder_dim
            if self.use_tf and self.training:
                lstm_input = torch.cat((embedding[:, t], gated_context), dim=1)
                #lstm_input = batch * (encoder_dim + embedding_dim)
            else:
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding, gated_context), dim=1)

            h, c = self.lstm(lstm_input, (h, c))
            # h = batch * hidden_size
            # c = batch * hidden_size
            output = self.deep_output(self.dropout(h))

            preds[:, t] = output
            alphas[:, t] = alpha

            if not self.training or not self.use_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size, 1))
        return preds, alphas

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c
    

    def caption(self, img_features, beam_size=17):
        """
        We use beam search to construct the best sentences following a
        similar implementation as the author in
        https://github.com/kelvinxu/arctic-captions/blob/master/generate_caps.py
        """
        prev_words = torch.zeros(beam_size, 1).long()

        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1)
        alphas = torch.ones(beam_size, 1, img_features.size(1))

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(img_features)

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words / output.size(1)
            next_word_idxs = top_words % output.size(1)

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > 50:
                break
            step += 1

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
        return sentence, alpha

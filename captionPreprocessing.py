import nltk
class CaptionProprocessor:
    def __init__(self, captions):
        self.captions = captions
        nltk.download('punkt')

    def tokenize(self, captions_list):
        list_of_tokens = set()
        max_caption_length = 0
        for caption in captions_list : 
            tokens = nltk.tokenize.word_tokenize(caption.lower()) 
            max_caption_length = max(max_caption_length, len(tokens))
            list_of_tokens.update(tokens)
        self.max_caption_length = max_caption_length
        return list_of_tokens, max_caption_length
    
    def preprocess(self, max_caption_length):
        padded_captions = []
        for caption in self.captions :
            padded_caption = self.padd_caption(caption, max_caption_length)
            padded_captions.append(padded_caption)
        return padded_captions
    
    def padd_caption(self, caption, max_caption_length):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        padded_tokens = tokens + ['<pad>'] * (max_caption_length - len(tokens))
        return padded_tokens
    
    def max_Caption_Length(self):
        return self.max_caption_length
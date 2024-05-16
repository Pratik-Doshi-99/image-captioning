import nltk
import Vocabulary
class CaptionProprocessor:
    def __init__(self):
        nltk.download('punkt')

    def tokenizeListOfCaptions(self, captions_list):
        list_of_tokens = set()
        max_caption_length = 0
        for caption in captions_list : 
            tokens = self.tokenizeCaption(caption.lower()) 
            max_caption_length = max(max_caption_length, len(tokens))
            list_of_tokens.update(tokens)
        self.max_caption_length = max_caption_length
        return list_of_tokens, max_caption_length
    
    def tokenizeCaption(self, caption):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        return tokens
    
    def preprocess(self, captions, max_caption_length):
        padded_captions = []
        for caption in captions :
            padded_caption = self.padd_caption(caption, max_caption_length)
            padded_captions.append(padded_caption)
        return padded_captions
    
    def padd_caption(self, caption, max_caption_length):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        tokens = ['<start>'] + tokens
        tokens = tokens + ['<end>']
        padded_tokens = tokens + ['<pad>'] * (max_caption_length - len(tokens))
        return padded_tokens
    
    def max_Caption_Length(self, captions_list):
        max_caption_length = 0
        for caption in captions_list : 
            tokens = self.tokenizeCaption(caption.lower()) 
            max_caption_length = max(max_caption_length, len(tokens))
        return max_caption_length
    
    def convertCaptionToIndices(self, caption, max_caption_length, vocab = Vocabulary.Vocabulary(vocab_file='vocab.pkl', vocab_from_file=True)):
        tokens = self.padd_caption(caption, max_caption_length)
        caption_indices = [vocab(token) for token in tokens]
        return caption_indices

        
        
import pickle
import os.path


class Vocabulary(object):

    def __init__(self,
        vocab_file='./vocab.pkl',
        tokenized_captions_list=[],
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        pad_word="<pad>",
        vocab_from_file=False):
        """Initialize the vocabulary.
        Args:
          vocab_file: File containing the vocabulary.
          start_word: Special word denoting sentence start.
          end_word: Special word denoting sentence end.
          unk_word: Special word denoting unknown words.
          pad_word: Special word denoting padding.
          vocab_from_file: If False, create vocab from scratch & override any existing vocab_file
                           If True, load vocab from from existing vocab_file, if it exists
        """
        self.vocab_file = vocab_file
        self.captions_list = tokenized_captions_list
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.pad_word = pad_word
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file OR build the vocabulary from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            print('-----Loading vocabulary from vocab.pkl file!----')
            with open(self.vocab_file, 'rb') as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print('----Vocabulary successfully loaded from vocab.pkl file!---')
        else:
            self.build_vocab()
            print('----Vocabulary successfully created from scratch!---')
            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self, f)
            print('----Vocabulary saved to vocab.pkl file!---')
        
    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_word(self.pad_word)
        self.add_tokens(self.captions_list)

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_tokens(self, tokens):
        """Add tokens from a set to the vocabulary."""
        for token in tokens:
            self.add_word(token)

    def __call__(self, word):
        """Convert a word to its corresponding index in the vocabulary"""
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        """Return the number of items in the vocabulary"""
        return len(self.word2idx)
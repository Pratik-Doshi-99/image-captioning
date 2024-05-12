import data 
import nltk 
import Vocabulary
import captionPreprocessing
caption_path = 'data/flickr8k_m/captions.txt'
print("----------Geeting Flickr8k dataset ---------------")

image_caption_pairs = data.get_flickr8k_captions(captions_path=caption_path)
captions_list = [inner_list[1] for inner_list in image_caption_pairs]

print("----------Tokenizing the captions ---------------")

caption_processor = captionPreprocessing.CaptionProprocessor(captions_list)
list_of_tokens, max_caption_length = caption_processor.tokenize(captions_list)
vocab_length = len(list_of_tokens)
print("Total Number of unique tokens :", vocab_length) 

print("-----------Tokenization Completed ---------------")

print("----------Creating Vocabulary ---------------")

vocab = Vocabulary.Vocabulary(vocab_file='vocab.pkl',
                                tokenized_captions_list=list_of_tokens,
                                vocab_from_file=True) # change it to False to create vocabulary from scratch
print("Total number of unique tokens in Vocabulary :" ,len(vocab))
assert len(vocab) == vocab_length + 4 # 4 is for start_word, end_word, unk_word, pad_word
# below assertion will fail if vocab_from_file is set to False since we are creating vocabulary from scratch
assert vocab('man') == 5779 
assert vocab('ayush') == 2 # unk_word
assert vocab('<pad>') == 3
assert vocab('<start>') == 0
assert vocab('<end>') == 1
assert vocab('<unk>') == 2
print("----------Vocabulary Creation Completed ---------------")


print("----------Let's add padding to these captions ---------------")

padded_captions = caption_processor.preprocess(max_caption_length)
print(padded_captions[0])
assert len(padded_captions[0]) == max_caption_length

print("----------Padding Completed ---------------")










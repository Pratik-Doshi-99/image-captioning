import data 
import Vocabulary
import captionPreprocessing
caption_path = 'data/flickr8k_m/captions.txt'
print("----------Geeting Flickr8k dataset ---------------")

image_caption_pairs = data.get_flickr8k_captions(captions_path=caption_path)
print(type (image_caption_pairs))
print(image_caption_pairs[0])
captions_list = [inner_list[1] for inner_list in image_caption_pairs]

print("----------Tokenizing the captions ---------------")

caption_processor = captionPreprocessing.CaptionProprocessor()
list_of_tokens, max_caption_length = caption_processor.tokenizeListOfCaptions(captions_list)
vocab_length = len(list_of_tokens)
print((vocab_length))
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

padded_captions = caption_processor.preprocess(captions= captions_list, max_caption_length= max_caption_length + 2 ) # +2 for start_word and end_word
print(padded_captions[0])
assert len(padded_captions[0]) == max_caption_length + 2 # +2 for start_word and end_word

print("----------Padding Completed ---------------")

print("----------Let's convert these image's captions to integers with padding ---------------")



caption_processor = captionPreprocessing.CaptionProprocessor()
image_caption_indices = []

for object in image_caption_pairs:
    image_name = object[0]
    caption = object[1]
    caption_indices = caption_processor.convertCaptionToIndices(caption= caption, max_caption_length= max_caption_length + 2)
    image_caption_indices.append([image_name, caption_indices])

print(image_caption_indices[0])
assert len(image_caption_indices[0][1]) == len(padded_captions[0]) 









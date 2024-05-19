import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataset import get_loader
import encoderDecoder
from torchtext.data.metrics import bleu_score

def indices_to_words(indices, vocab):
    decoded_sentences = []
    for idx_row in indices:
        sentence = []
        for idx in idx_row:
            word = vocab.get_word(idx.item())
            if(word == '<start>'):
                continue 
            if word == '<end>':  # Stop decoding if '<end>' token is encountered
                break
            sentence.append(word)
        decoded_sentences.append(sentence)
    return decoded_sentences

def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


    

    train_loader, train_dataset = get_loader(
        root_folder="data/flickr8k/Images",
        captions_file="data/flickr8k/captions_train.txt",
        transform=transform,
        num_workers=2,
    )

    val_dataset, val_loader = get_loader(
        root_folder="data/flickr8k/Images",
        captions_file="data/flickr8k/captions_val.txt",
        transform=transform,
        num_workers=2,
    )


    # torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else \
         ("mps" if torch.backends.mps.is_available() else "cpu" ) 
    
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(train_dataset.vocab)
    num_layers = 3
    learning_rate = 0.001
    num_epochs = 10
    drop_prob = 0.5

    # initialize model, loss etc
    model = encoderDecoder.EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers, drop_prob ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.get_index("<pad>") )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        loss = 0 
        for image, caption in train_loader:
            image = image.to(device)
            caption = caption.to(device)

            output = model(image, caption)
            adjusted_output = output[:, :caption.size(1), :]  # Exclude first token to align sequence lengths

            # Flatten the adjusted output tensor and the captions tensor
            flatten_output = adjusted_output.reshape(-1, adjusted_output.shape[2])
            flatten_captions = caption.reshape(-1)

            loss = criterion(flatten_output, flatten_captions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        


if __name__ == "__main__":
    train()
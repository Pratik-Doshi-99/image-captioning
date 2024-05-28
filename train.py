import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from dataset import get_loader
import encoderDecoder
from torchtext.data.metrics import bleu_score


def indices_to_words(indices, vocab, referenceCorpus):
    decoded_sentences = []
    count = 0 ; 
    for idx_row in indices:
        count += 1 
        sentence = [] 
        for idx in idx_row:
            word = vocab.get_word(idx.item())
            if(word == '<start>'):
                continue 
            if word == '<end>':  # Stop decoding if '<end>' token is encountered
                break
            sentence.append(word)
        if(referenceCorpus):
            corpus = []
            corpus.append(sentence)
            decoded_sentences.append(corpus)
        else: 
            decoded_sentences.append(sentence)
    return decoded_sentences

def train():
    torch.manual_seed(42)

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataset, train_loader = get_loader(
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
    drop_prob = 0.4

    # initialize model, loss etc
    model = encoderDecoder.EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers, drop_prob, train_CNN= train_CNN ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.get_index("<pad>") )
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        loss = 0 
        for image, caption in train_loader:
            image = image.to(device)
            caption = caption.to(device)
            output = model(image, caption)
            loss = criterion(output.view(-1, vocab_size), caption.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #validation 
        model.eval()
        val_loss= 0.0 
        bleu_score_value = 0.0 
        with torch.no_grad(): 
            for image, caption in val_loader:
                image = image.to(device)
                caption = caption.to(device)
                ground_truth_captions = indices_to_words(caption, val_dataset.vocab, True)

                output = model(image, caption)
                _, predicted_indices = torch.max(output, dim=2)
                predicted_captions = indices_to_words(predicted_indices, val_dataset.vocab, False)
                bleu_score_value += bleu_score(predicted_captions, ground_truth_captions, max_n=2, weights=[0.6, 0.4])
                val_loss += criterion(output.view(-1, vocab_size), caption.view(-1))
            
        val_loss /= len(val_loader)
        bleu_score_value /= len(val_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {val_loss:.4f}, Accuracy: {bleu_score_value:.4f}')
        
        # Save the model with least loss 
        if(epoch == 0):
            min_loss = val_loss
        else:
            if(val_loss < min_loss):
                min_loss = val_loss
                torch.save(model.state_dict(), "model.pth")
                print("Model saved")




if __name__ == "__main__":
    train()
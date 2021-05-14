import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

from tqdm import tqdm
from collections import Counter
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
 
import torchvision.models as models
import torchvision.transforms as T
 
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchvision
from PIL import Image

import PIL
from time import time
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor, RandomHorizontalFlip, RandomCrop

from Levenshtein import distance as levenshtein_distance


df = pd.read_csv("data.csv").head(150)



# sizes = []
# for idx, row in tqdm(df[1:1000].iterrows()):
#   s = time()
#   path = row['original_path']
#   e = time()
#   print("row: ",(e - s))
#   s = time()
#   size = os.path.getsize('data/'+ path)
#   e = time()
#   print("os path getsize: ",(e - s))
  

print("Unique InChI: ",df.InChI.nunique())
df.InChI.count()

df[:100].InChI

df.head()

# root = "data/"
# print(root+df["original_path"].iloc[0])
# jpgfile = mpimg.imread(root+df["original_path"].iloc[0])

# plt.imshow(jpgfile, cmap="gray")

# x = np.asarray(jpgfile)
# print(x.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Vocabulary:
  def __init__(self,freq_threshold):
      #setting the pre-reserved tokens int to string tokens
      
      #string to int tokens
      #its reverse dict self.itos
      self.stoi={'C': 0,')': 1,'P': 2,'l': 3,'=': 4,'3': 5,'N': 6,'I': 7,'2': 8,'6': 9,'H': 10,'4': 11,'F': 12,'0': 13,'1': 14,'-': 15,'O': 16,'8': 17,
  ',': 18,'B': 19,'(': 20,'7': 21,'r': 22,'/': 23,'m': 24,'c': 25,'s': 26,'h': 27,'i': 28,'t': 29,'T': 30,'n': 31,'5': 32,'+': 33,'b': 34,'9': 35,
  'D': 36,'S': 37,'<SOS>': 38,'<EOS>': 39,'<PAD>': 40}
      
      self.itos = {item[1]:item[0] for item in self.stoi.items()}
      self.freq_threshold = freq_threshold
      
  def __len__(self): return len(self.itos)

  @staticmethod
  def tokenize(text):
      return [char for char in text]

  def build_vocab(self, sentence_list):
      frequencies = Counter()
      idx = 4
      
      for sentence in sentence_list:
          for word in self.tokenize(sentence):
              frequencies[word] += 1
              
              #add the word to the vocab if it reaches minum frequecy threshold
              if frequencies[word] == self.freq_threshold:
                  self.stoi[word] = idx
                  self.itos[idx] = word
                  idx += 1

  def numericalize(self,text):
      """ For each word in the text corresponding index token for that word form the vocab built as list """
      tokenized_text = self.tokenize(text)
      return [ self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text ] 


class MoleculesDataset(Dataset):
  def __init__(self, csv_file, transform):
      self.df = pd.read_csv(csv_file).head(150)
      self.root = "./data/"
      self.transform = transform
      
      self.vocab = Vocabulary(0)
      # self.vocab.build_vocab(self.df["InChI"].tolist())
      
  def __len__(self):
      return len(self.df)

  def __getitem__(self, idx):
      # print(f'Getting item with idx: {idx}')
      row = self.df.iloc[idx]
      # s = time()
      tensorImage = torchvision.transforms.functional.to_tensor(Image.open(self.root+row["original_path"]))
      # e = time()
      # print("tensorImage takes: ", (e-s))
      caption_vec = []
      # s = time()
      caption_vec += [self.vocab.stoi["<SOS>"]]
      # e = time()
      # print("caption_vec SOS: ", (e-s))
      # s = time()
      caption_vec += self.vocab.numericalize(row["InChI"])
      # e = time()
      # print("caption_vec InChI: ", (e-s))
      # s = time()
      caption_vec += [self.vocab.stoi["<EOS>"]]
      # e = time()
      # print("caption_vec EOS: ", (e-s))

      return (
          self.transform(tensorImage),
          torch.as_tensor(caption_vec)
      )


transform = Compose([
    #RandomHorizontalFlip(),
    Resize((256,256)),
    #ToTensor(),
    Normalize(mean=[0.5], std=[0.5]),
    ])

dataset = MoleculesDataset("data.csv", transform)

pad_idx = dataset.vocab.stoi["<PAD>"]

class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self,pad_idx,batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self,batch):
        # print('Inside collate')
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        # print('Collate finished')
        return imgs,targets



class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        for param in resnet.parameters():
            param.requires_grad_(False)
            
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.permute(0, 2, 3, 1)
        features = features.view(features.size(0), -1, features.size(-1))
        # print("Shape of features: ",features.shape)
        return features

#Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        
        self.attention_dim = attention_dim
        
        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)
        
        self.A = nn.Linear(attention_dim,1)
        
    def forward(self, features, hidden_state):
        u_hs = self.U(features)
        w_ah = self.W(hidden_state)
        
        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))
        
        attention_scores = self.A(combined_states)
        attention_scores = attention_scores.squeeze(2)
        
        alpha = F.softmax(attention_scores, dim=1)
        
        attention_weights = features*alpha.unsqueeze(2)
        attention_weights = attention_weights.sum(dim=1)
        
        return alpha, attention_weights
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        
        self.lstm_cell = nn.LSTMCell(embed_size+encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        
        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(drop_prob)
        
    def forward(self, features, captions):
        embeds = self.embedding(captions)
        
        #initialize LSTM state
        h, c = self.init_hidden_state(features) #(batch_size, decoder_dim)
        
        #get the seq length to iterate
        seq_length = len(captions[0])-1
        batch_size = captions.size(0)
        num_features = features.size(1)
        
        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(device)
        
        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h,c = self.lstm_cell(lstm_input, (h,c))
            
            output = self.fcn(self.dropout(h))
            
            preds[:, s] = output
            alphas[:, s] = alpha
            
        return preds, alphas
                
    def generate_caption(self, features, max_length=20, vocab=None):
        
        batch_size = features.size(0)
        h, c = self.init_hidden_state(features)
        
        alphas=[]
        
        word = torch.tensor(vocab.stoi['<SOS>']).view(1,-1).to(device)
        embeds = self.embedding(word)

        #??
        captions=[]
        
        for i in range(max_length):
            alpha, context = self.attention(features, h)
            
            alphas.append(alpha.cpu().detach().numpy())
            
            lstm_input = torch.cat((embeds[:, 0], context), dim=1)
            h,c = self.lstm_cell(lstm_input, (h,c))
            output = self.fcn(self.dropout(h))
            output = output.view(batch_size,-1)
            
            #select the word
            predicted_word_idx = output.argmax(dim=1)
            
            #save the generated word
            captions.append(predicted_word_idx.item())
            
            if predicted_word_idx.item() == vocab.stoi["<EOS>"]:
                break
            
            #send generated word as the next caption
            embeds = self.embedding(predicted_word_idx.unsqueeze(0))
            
        return [vocab.itos[idx] for idx in captions],alphas

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        
        return h,c


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim, drop_prob=0.3):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim
        )
        
    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        
        return outputs
        


#Hyperparams
embed_size=200
vocab_size = len(dataset.vocab)
attention_dim=300
encoder_dim=2048
decoder_dim=300

model = EncoderDecoder(
    embed_size=embed_size,
    vocab_size=vocab_size,
    attention_dim=attention_dim,
    encoder_dim=encoder_dim,
    decoder_dim=decoder_dim
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr = 3e-4)


#helper function to save the model
def save_model(model,num_epochs):
    model_state = {
        'num_epochs':num_epochs,
        'embed_size':embed_size,
        'vocab_size':len(dataset.vocab),
        'attention_dim':attention_dim,
        'encoder_dim':encoder_dim,
        'decoder_dim':decoder_dim,
        'state_dict':model.state_dict()
    }

    torch.save(model_state,'attention_model_state_150.pth')


num_epochs=2000
# print_every=50
iteration = 0

train_losses = []
test_losses = []
lev = []

distances = {}
losses = {}

train_dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=8, 
    shuffle=True,
    num_workers=0, 
    collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
)
test_dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=False,
    num_workers=0, 
    collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
)

dataloader = {"train" : train_dataloader, "test" : test_dataloader}

for epoch in tqdm(range(1, num_epochs+1)):
    print("Epoch: ", epoch)
    
    for phase in ["train", "test"]:
        print(f"Currently in phase {phase} !")

        for image, captions in tqdm(dataloader[phase]):

            #imageTensor, captions = imageTensor.to(device), captions.to(device)
            image, captions = image.to(device), captions.to(device)

            optimizer.zero_grad()
            
            outputs, attentions = model(image, captions)
            targets = captions[:, 1:]
            
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            train_losses.append(loss)
            
            if phase == "train":
                loss.backward()
                optimizer.step()

            if phase == "test":
                # if (iteration + 1) % print_every == 0:
                # print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))
                test_losses.append(loss)
                
                model.eval()
                
                with torch.no_grad():

                    features = model.encoder(image[0:1].to(device))
                    caps, alphas = model.decoder.generate_caption(features, vocab=dataset.vocab, max_length=embed_size)

                    caption_predicted = ''.join(caps)
                    
                    caption_target = captions.cpu().numpy().tolist()[0][1:]
                    caption_target = ''.join([dataset.vocab.itos[idx] for idx in caption_target if idx !=40])
                    
                    print("caption_target object: \n", caption_target)
                    print(caption_predicted)

                    levenshtein_metric = levenshtein_distance(caption_target, caption_predicted)
                    lev.append(float(levenshtein_metric))
                    print(levenshtein_metric)
                    print(lev)
                    print(np.array(lev).sum() / len(lev))
        
        if phase == "test":
            print("Printing metrics!")
            distances[epoch] = np.array(lev).sum() / len(lev)
            losses[epoch] = np.array(test_losses).sum() / len(test_losses)

            print(f"The average levenstein distance for the wbole dataset for epoch {epoch} is : ", distances[epoch])

            model.train()
        
            
    save_model(model, epoch)

print(train_losses)
img, caption = dataset[0]
print("Sentence:")
print(''.join([dataset.vocab.itos[token] for token in caption.tolist()]))
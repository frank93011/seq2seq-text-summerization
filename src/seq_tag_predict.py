import os
import pickle
from argparse import Namespace
from typing import Tuple, Dict
from torch import optim
from tqdm import tnrange
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import SeqTaggingDataset
from utils import Tokenizer
from dataset import SeqTaggingDataset
import numpy as np
import json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--output_path')
args = parser.parse_args()
OUT_DATA_PATH = args.output_path

with open("./datasets/seq_tag/test.pkl", "rb") as f:
    valid = pickle.load(f)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self,
                 embedding_path,
                 embed_size,
                 rnn_hidden_size) -> None:
        super(Encoder, self).__init__()
        with open(embedding_path, 'rb') as f:
          embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.rnn = nn.LSTM(embed_size, rnn_hidden_size, 1, bidirectional=True)
        self.proj = nn.Linear(rnn_hidden_size*2, 1)
        self.dropout = nn.Dropout(0.5)
        
        # init a LSTM/RNN

    def forward(self, idxs):
        embed = self.embedding(idxs)
        output, state = self.rnn(embed)
        output = self.dropout(output)
        out = torch.stack([self.proj(output[t]) for t in range(output.size(0))])
        return out

def train(model, train_data, val_data, hparams, criterion):
    train_loader = DataLoader(train_data, hparams.batch_size, collate_fn=train_data.collate_fn, shuffle=True)
    
    pt = './save/seq2tag.pt'
    optimizer =  optim.Adam(model.parameters(), lr = 0.0001)
    total_step = len(train_loader)
    
    tr_loss = 0.
    min_val_loss = 1000
    
    iters = 0
    pos_weight = torch.tensor([hparams.pos_weight], device=device)
    for epoch in tnrange(100):
        tr_loss = 0.
        for idx, batch in enumerate(train_loader):
            iters += 1
            text = batch['text'].t().to(device)
            label = batch['label'].t().type(torch.FloatTensor).to(device)
            mask = label.ne(-1) # mask out pad
            model.train()
            outputs = model(text).squeeze(-1)
            outputs = torch.masked_select(outputs, mask)
            label = torch.masked_select(label, mask)
            loss = criterion(outputs, label) #(hidden, batch)
#             print(loss.size(), label.size())
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss = loss.detach().cpu().item() 
            
            tr_loss += loss 
        
        tr_loss /= len(train_loader)
        val_loss = evaluate(model, val_data, hparams, criterion)
        
        print(f'[Epoch {epoch+1}] loss: {tr_loss:.3f} '+
              f'val_loss: {val_loss:.3f}', flush=True)
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), pt)
            
def evaluate(model, valid, hparams, criterion):
    val_loader = DataLoader(valid, batch_size=hparams.batch_size, collate_fn=valid.collate_fn)
    v_loss = 0.
    n_acc = 0.
    total_steps = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            text = batch['text'].to(device)
            label = batch['label'].type(torch.FloatTensor).to(device)
            mask = label.ne(-1)
            outputs = model(text).squeeze(-1)
            outputs = torch.masked_select(outputs, mask)
            label = torch.masked_select(label, mask)
            loss = criterion(outputs, label)
            loss = loss.mean()
            v_loss += loss.detach().cpu().item()
            
    return v_loss / len(val_loader)

hparams = Namespace(**{
    'embedding_path': "./datasets/seq2seq/embedding.pkl",
    'embed_size': 300,
    'ignore_idx': -100,

    'batch_size': 64,
    'pos_weight': 10,

    'rnn_hidden_size': 128,
})

encoder = Encoder(hparams.embedding_path, hparams.embed_size,hparams.rnn_hidden_size).to(device)
# encoder = Encoder(hparams.embedding_path, hparams.embed_size,hparams.rnn_hidden_size)
criterion = nn.BCEWithLogitsLoss(reduction='none',pos_weight=torch.tensor(hparams.pos_weight))
# encoder

# Encoder(
#   (embedding): Embedding(97513, 300)
#   (rnn): GRU(300, 256, bidirectional=True)
#   (fc): Linear(in_features=512, out_features=1, bias=True)
# )

def predict_topk(model, valid, hparams, top_k=2):
    val_loader = DataLoader(valid, batch_size=1, collate_fn=valid.collate_fn)
    out = []
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            x = batch['text'].to(device)
            if len(x[0]) == 0: # if empty
                out.append({"id":batch['id'][0], "predict_sentence_index": [0]}) + '\n'
                continue
            pred = torch.sigmoid(model(x)).squeeze(-1) # turn output to probability
            pred = pred[0] > 0.5
            ext = []
            range_tup = batch['sent_range'][0]
            for tup in range_tup:
                ext.append(float(sum(pred[tup[0]:tup[1]])) / (tup[1]-tup[0]+1))
            
            topk = min(top_k, len(ext))
            ans = np.array(ext).argsort()[-topk:][::-1].tolist()
            
            if len(ext) == 0:
                ext.append(0)
            else:
                ext = ans

            out.append({"id":batch['id'][0], "predict_sentence_index": ext})
    
    return out

def predict_out(list_dict, file_path):
    with open(file_path , 'w') as outfile:
        for entry in list_dict:
            json.dump(entry, outfile)
            outfile.write('\n')

prediction = predict_topk(encoder, valid, hparams, 2)
predict_out(prediction, OUT_DATA_PATH)
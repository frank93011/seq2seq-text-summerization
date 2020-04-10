import math
import torch
import random
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_size, embedding, embedding_dim, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        emb = nn.Embedding(input_size, embedding_dim)
        emb.weight.data.copy_(embedding.vectors)
        self.embedding = emb
        self.embedding.weight.requires_grad = False
        self.rnn = nn.GRU(300, hidden_size, n_layers, dropout = dropout ,bidirectional=True)
        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, src, hidden=None):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # sum bidirectional outputs
#         outputs = (outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:])
        hidden = torch.tanh(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
#         print(hidden)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding, embedding_dim, enc_hidden_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hidden_size = hid_dim
        self.n_layers = n_layers
        self.emb_dim = embedding_dim
    
        emb = nn.Embedding(output_dim, embedding_dim)
        emb.weight.data.copy_(embedding.vectors)
        self.embedding = emb
        
        self.rnn = nn.GRU(embedding_dim, hid_dim, dropout=0.5)
        
        self.fc_out = nn.Linear(hid_dim, output_dim , bias=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, hidden = self.rnn(embedded, hidden)
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        prediction = self.fc_out(output.squeeze(0))
        #prediction = [batch size, output dim]
        return prediction, hidden



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
#         assert encoder.hidden_size == decoder.hidden_size, \
#             "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
#         print(src)
        encoder_outputs, hidden = self.encoder(src)
        hidden = hidden.unsqueeze(0)
#         hidden = hidden.view(self.encoder.n_layers, batch_size, -1)
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(input, hidden)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs
    
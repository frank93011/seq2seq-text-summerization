import os
import json
import random
import pickle
import math
from pathlib import Path
from utils import Tokenizer, Embedding
from torch.nn.utils import clip_grad_norm
from dataset import Seq2SeqDataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from argparse import ArgumentParser
from attention_model import Encoder, Attention, Seq2Seq, Decoder


parser = ArgumentParser()
parser.add_argument('--output_path')
args = parser.parse_args()
OUT_DATA_PATH = args.output_path

TEST_DATA_PATH = './datasets/seq2seq/test.pkl'
with open(TEST_DATA_PATH, "rb") as f:
    test = pickle.load(f)
with open("./datasets/seq2seq/embedding.pkl", "rb") as f:
    embedding = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
en_size = len(embedding.vocab)
hidden_size = 300
lr = 0.001
encoder = Encoder(en_size, embedding, hidden_size,
                  n_layers=2, dropout=0.2)
decoder = Decoder(embedding, 300, hidden_size, en_size)
seq2seq = Seq2Seq(encoder, decoder).cuda()

optimizer = optim.Adam(seq2seq.parameters(), lr)
print(seq2seq)
seq2seq.load_state_dict(torch.load('./src/save/attention_model.pt'))

SOS_token = 1
EOS_token = 2

def generate_pair(data):
    pairs = []
    input_length = len(data.data)
    for i in range(input_length):
        input = data.__getitem__(i)['text']
        id = data.__getitem__(i)['id']
        pairs.append((input, id))
    return pairs


class Sentence:
    def __init__(self, decoder_hidden, last_idx=SOS_token, sentence_idxes=[], sentence_scores=[]):
        if(len(sentence_idxes) != len(sentence_scores)):
            raise ValueError("length of indexes and scores should be the same")
        self.decoder_hidden = decoder_hidden
        self.last_idx = last_idx
        self.sentence_idxes =  sentence_idxes
        self.sentence_scores = sentence_scores

    def avgScore(self):
        if len(self.sentence_scores) == 0:
            raise ValueError("Calculate average score of sentence, but got no word")
        # return mean of sentence_score
        return sum(self.sentence_scores) / len(self.sentence_scores)

    def addTopk(self, topi, topv, decoder_hidden, beam_size, voc):
        topv = torch.log(topv)
        terminates, sentences = [], []
        for i in range(beam_size):
            if topi[0][i] == EOS_token:
                terminates.append(([voc[idx.item()] for idx in self.sentence_idxes] + ['<EOS>'],
                                   self.avgScore())) # tuple(word_list, score_float
                continue
            idxes = self.sentence_idxes[:] # pass by value
            scores = self.sentence_scores[:] # pass by value
            idxes.append(topi[0][i])
            scores.append(topv[0][i])
            sentences.append(Sentence(decoder_hidden, topi[0][i], idxes, scores))
        return terminates, sentences

    def toWordScore(self, voc):
        words = []
        for i in range(len(self.sentence_idxes)):
            if self.sentence_idxes[i] == EOS_token:
                words.append('<EOS>')
            else:
                words.append(voc[self.sentence_idxes[i].item()])
        if self.sentence_idxes[-1] != EOS_token:
            words.append('<EOS>')
        return (words, self.avgScore())

def decode(decoder, decoder_hidden, input_len, encoder_outputs, voc, max_length=40):

    decoder_input = torch.LongTensor([SOS_token])
    decoder_input = decoder_input.to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, input_len) #TODO: or (MAX_LEN+1, MAX_LEN+1)
#     print(decoder_hidden.size())
#     all_attention = []
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        _, topi = decoder_output.topk(3)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(voc[ni.item()])

        decoder_input = torch.LongTensor([ni])
        decoder_input = decoder_input.to(device)

    return decoded_words, decoder_attentions[:di + 1]
#     return decoded_words, decoder_attentions


def evaluate(encoder, decoder, voc, sentence , max_length=40):
    indexes_batch = [sentence] #[1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    batch_size = input_batch.size(1)
    encoder_outputs, encoder_hidden = encoder(input_batch, None)

    decoder_hidden = encoder_hidden[:1]
#     decoder_hidden = encoder_hidden.view(encoder.n_layers, batch_size, -1)
    return decode(decoder, decoder_hidden, lengths[0], encoder_outputs, voc)


def evaluateRandomly(encoder, decoder, tokenizer, voc, pairs, reverse, beam_size, n=10):
    for _ in range(n):
        pair = random.choice(pairs)
        print("=============================================================")
        if reverse:
            print('>', " ".join(reversed(pair[0].split())))
        else:
            print('>', tokenizer.decode(pair[0]))
            print('=', tokenizer.decode(pair[1]))
        if beam_size == 1:
            output_words, _ = evaluate(encoder, decoder, voc, pair[0], beam_size)
            output_sentence = ' '.join(output_words)
            print('<', output_sentence)
        else:
            output_words_list = evaluate(encoder, decoder, voc, pair[0], beam_size)
            for output_words, score in output_words_list:
                output_sentence = ' '.join(output_words)
                print("{:.3f} < {}".format(score, output_sentence))
                
def predict_out(list_dict, file_path):
    with open(file_path , 'w') as outfile:
        for entry in list_dict:
            json.dump(entry, outfile)
            outfile.write('\n')

def predict(encoder, decoder, voc, pairs):
    n = len(pairs)
    out = []
    show_per = 100
    for i in range(n):
        predict = {}
        pair = pairs[i]
        output_words, attention = evaluate(encoder, decoder, voc, pair[0])
#         atten = torch.Tensor.cpu(attention.detach()).squeeze(0)
#         atten = atten
        output_sentence = ' '.join(output_words)
        predict['id'] = pair[1]
        predict['predict'] = output_sentence[:-6]
        out.append(predict)
    return out

tokenizer = Tokenizer(embedding.vocab, lower=False)
pairs = generate_pair(test)
out = predict(seq2seq.encoder, seq2seq.decoder, embedding.vocab, pairs)
predict_out(out, OUT_DATA_PATH)
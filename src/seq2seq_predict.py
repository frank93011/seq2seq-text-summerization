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
from seq2seq_model import Encoder, Decoder, Seq2Seq

parser = ArgumentParser()
parser.add_argument('--output_path')
args = parser.parse_args()
OUT_DATA_PATH = args.output_path

with open("./datasets/seq2seq/test.pkl", "rb") as f:
    test = pickle.load(f)
with open("./datasets/seq2seq/embedding.pkl", "rb") as f:
    embedding = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = len(embedding.vocab)
OUTPUT_DIM = len(embedding.vocab)
ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
ENC_HID_DIM = 128
DEC_HID_DIM = 256
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, embedding,ENC_EMB_DIM, ENC_HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, embedding, ENC_EMB_DIM,ENC_HID_DIM, DEC_HID_DIM, N_LAYERS, DEC_DROPOUT)
model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load('./src/save/seq2seq_model.pt'))

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

def decode(decoder, decoder_hidden, encoder_outputs, voc, max_length=40):

    decoder_input = torch.LongTensor([SOS_token])
    decoder_input = decoder_input.to(device)

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length) #TODO: or (MAX_LEN+1, MAX_LEN+1)
#     print(decoder_hidden.size())

    for di in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
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


def evaluate(encoder, decoder, voc, sentence , max_length=40):
    indexes_batch = [sentence] #[1, seq_len]
    lengths = [len(indexes) for indexes in indexes_batch]
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    batch_size = input_batch.size(1)
    encoder_outputs, encoder_hidden = encoder(input_batch, None)

    decoder_hidden = encoder_hidden.view(encoder.n_layers, batch_size, -1)
    return decode(decoder, decoder_hidden, encoder_outputs, voc)


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
    show_per = 4000
    for i in range(n):
        predict = {}
        pair = pairs[i]
        output_words, attention = evaluate(encoder, decoder, voc, pair[0])
        output_sentence = ' '.join(output_words)
        # if(i % show_per == 0):
        #     print('>', tokenizer.decode(pair[0]))
        #     print('<', output_sentence[:-6])
#             show_attention(tokenizer.decode(pair[0]), output_sentence, attention)
        predict['id'] = pair[1]
        predict['predict'] = output_sentence[:-6]
        out.append(predict)
    return out

tokenizer = Tokenizer(embedding.vocab, lower=False)
pairs = generate_pair(test)
out = predict(model.encoder, model.decoder, embedding.vocab, pairs)
predict_out(out, OUT_DATA_PATH)
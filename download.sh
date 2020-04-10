#!/bin/bash
wget https://www.dropbox.com/s/ru3n8ie4yqc38dd/seq2seq_embedding.pkl?dl=1 -O ./datasets/seq2seq/embedding.pkl
wget https://www.dropbox.com/s/6lub92iiok6q3ni/seq_tag_embedding.pkl?dl=1 -O ./datasets/seq_tag/embedding.pkl
wget https://www.dropbox.com/s/7tkolu7bj68zxtq/seq2tag-final.pt?dl=1 -O ./src/save/seq2tag-final.pt
wget https://www.dropbox.com/s/883fl91ygarg11n/attention_model.pt?dl=1 -O ./src/save/attention_model.pt
wget https://www.dropbox.com/s/041jk0rw71k6zt3/seq2seq_model.pt?dl=1 -O ./src/save/seq2seq_model.pt

bash install_packages.sh
#!/usr/bin/env bash

python3.6 ./src/preprocess_seq2seq_test.py --test_data_path $1
python3.6 ./src/attention_predict.py --output_path $2
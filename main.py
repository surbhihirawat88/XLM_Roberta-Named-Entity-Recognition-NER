import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd

"""BERT NER Inference."""


import json
import os

import torch
import torch.nn.functional as F
from transformers import XLMRobertaTokenizer,XLMRobertaForTokenClassification,RobertaConfig

from tqdm import tqdm, trange
import json
import io

app = Flask(__name__)

tag1=['I-art',
 'I-eve',
 'B-eve',
 'I-gpe',
 'I-geo',
 'B-nat',
 'I-per',
 'B-gpe',
 'I-org',
 'B-tim',
 'B-org',
 'I-tim',
 'I-nat',
 'B-art',
 'O',
 'B-per',
 'B-geo',
 'PAD']
# The name of the folder containing the model files.
PATH = "ner12.pt"

tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base" )


def generate_ner(context):
    tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
    model = torch.load(PATH, map_location='cpu')
    tokenized_sentence = tokenizer.encode(context)
    input_ids = torch.tensor([tokenized_sentence])
    with torch.no_grad():
        output = model(input_ids)
    label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
    new_tokens, new_labels = [], []
    for token, label_idx in zip(tokens, label_indices[0]):
        if token.startswith("##"):
            new_tokens[-1] = new_tokens[-1] + token[2:]
        else:
            new_labels.append(tag1[label_idx])
            new_tokens.append(token)
    prediction = [{"word": token, "tag": label} for token, label in zip(new_tokens, new_labels)]
    return prediction

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/demon', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        cont = request.form['context']
        pred= generate_ner(cont)
        return render_template('index.html', data=pred)


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=8001, debug=True)
    app.run(debug=True)  # running the app
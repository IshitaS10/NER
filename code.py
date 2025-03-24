# -*- coding: utf-8 -*-

import numpy as np # linear algalgebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pickle

import random

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/content/drive/MyDrive/train_data (1).pkl'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pickle.load(open('/content/drive/MyDrive/train_data (1).pkl','rb'))
train_data[0]

!pip install spacy==3.1.3

import spacy
import random
nlp = spacy.blank('en')

def train_model(train_data):
    # Remove all pipelines and add NER pipeline from the model
    if 'ner'not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        # adding NER pipeline to nlp model
        nlp.add_pipe(ner,last=True)

    #Add labels in the NLP pipeline
    for _, annotation in train_data:
        for ent in annotation.get('entities'):
            ner.add_label(ent[2])

    #Remove other pipelines if they are there
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(10): # train for 10 iterations
            print("Starting iteration " + str(itn))
            random.shuffle(train_data)
            losses = {}
            index = 0
            for text, annotations in train_data:
                try:
                    nlp.update(
                        [text],  # batch of texts
                        [annotations],  # batch of annotations
                        drop=0.2,  # dropout - make it harder to memorise data
                        sgd=optimizer,  # callable to update weights
                        losses=losses)
                except Exception as e:
                    pass

            print(losses)
# Start Training model
train_model(train_data)

from google.colab import drive
drive.mount('/content/drive')

# Saving the model
nlp.to_disk('nlp_ner_model')

#Loading Model
nlp_model = spacy.load('nlp_ner_model')

# trying and seeing the prediction of the model
doc = nlp_model(train_data[0][0])
for ent in doc.ents:
    print(f"{ent.label_.upper():{30}}-{ent.text}")

!pip install PyMuPDF
import fitz

import sys
fname = ''
doc = fitz.open(fname)
text = ""
for page in doc:
    text += page.get_text()

tx = " ".join(text.split('\n'))
print(tx)

# Applying the model
doc = nlp_model(tx)
for ent in doc.ents:
    print(f'{ent.label_.upper():{30}}- {ent.text}')

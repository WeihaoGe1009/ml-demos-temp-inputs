# this code pre-trains markov train for direct load

import re
from collections import defaultdict
import pickle
import nltk
from nltk.tokenize import sent_tokenize
import os

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def build_phrase_markov_chain(corpus_dir, order=2):
    chain = defaultdict(list)
    sentence_bank = []  # keep all sentences for prompt matching

    for filename in os.listdir(corpus_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(corpus_dir, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
                sentences = sent_tokenize(text)
                sentence_bank.extend(sentences)

                # Build transitions
                for i in range(len(sentences) - order):
                    key = tuple(sentences[i:i+order])
                    next_sent = sentences[i + order]
                    chain[key].append(next_sent)

    return chain, sentence_bank

chain, sentence_bank = build_phrase_markov_chain("../data/corpus_raw", order=2)

with open("../data/markov_model.pkl", "wb") as f:
    pickle.dump((chain, sentence_bank), f)

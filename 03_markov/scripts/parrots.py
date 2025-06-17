# this is the code containing Parrot_Search and Parrot_Markov
# We will compare retrieving info and generating info 

import bisect
import pickle
import numpy as np
import random
from collections import defaultdict, Counter
import re

### Parrot_Search

class ParrotSearchBWT:
    def __init__(self, bwt_input_bytes_path, sa_path, paragraph_offsets_path, paragraph_map_path):
        # Load original corpus and SA
        with open(bwt_input_bytes_path, "rb") as f:
            self.bwt_input_bytes = pickle.load(f)
        with open(sa_path, "rb") as f:
            self.sa = pickle.load(f)

        # Build BWT string from corpus and suffix array
        self.bwt = bytes([
            self.bwt_input_bytes[i - 1] if i != 0 else self.bwt_input_bytes[-1]
            for i in self.sa
        ])
        self.length = len(self.bwt)

        # Load paragraph structure
        with open(paragraph_offsets_path, "rb") as f:
            self.paragraph_offsets = pickle.load(f)
        with open(paragraph_map_path, "rb") as f:
            self.paragraph_map = pickle.load(f)

        # Build FM-index
        self.C = self._build_C()
        self.Occ = self._build_Occ()

    def _build_C(self):
        freqs = Counter(self.bwt)
        chars = sorted(freqs)
        C = {}
        total = 0
        for c in chars:
            C[c] = total
            total += freqs[c]
        return C

    def _build_Occ(self):
        unique_chars = sorted(set(self.bwt))
        Occ = {c: np.zeros(self.length + 1, dtype=np.uint32) for c in unique_chars}
        for i, c in enumerate(self.bwt):
            for k in Occ:
                Occ[k][i + 1] = Occ[k][i]
            Occ[c][i + 1] += 1
        return Occ

    def _range(self, pattern: bytes):
        if not pattern:
            return (0, self.length)

        i = len(pattern) - 1
        c = pattern[i]
        if c not in self.C:
            return (0, 0)

        l = self.C[c]
        r = self.C[c] + self.Occ[c][self.length]

        while i > 0 and l < r:
            i -= 1
            c = pattern[i]
            if c not in self.C:
                return (0, 0)
            l = self.C[c] + self.Occ[c][l]
            r = self.C[c] + self.Occ[c][r]
        return l, r

    def search(self, keyword: str):
        pattern = keyword.encode("utf-8")
        l, r = self._range(pattern)
        verified_positions = []

        for i in range(l, r):
            pos = self.sa[i]
            #  Safe match check using real corpus
            if self.bwt_input_bytes[pos:pos + len(pattern)] == pattern:
                verified_positions.append(pos)

        seen_paragraphs = set()
        results = []
        for pos in verified_positions:
            para_idx = bisect.bisect_right(self.paragraph_offsets, pos) - 1
            if para_idx >= 0:
                offset = self.paragraph_offsets[para_idx]
                if offset not in seen_paragraphs:
                    seen_paragraphs.add(offset)
                    results.append(self.paragraph_map[offset])
        return results


### helper functions for Parrot Search to perform search and polish outputs
def multi_keyword_search(parrot, keywords):
    """
    Perform AND-based multi-keyword search.
    Returns only paragraphs that contain *all* keywords.
    """
    matched_sets = []

    for kw in keywords:
        kw = kw.lower()
        result = parrot.search(kw)
        matched_offsets = {
            parrot.paragraph_offsets[bisect.bisect_right(parrot.paragraph_offsets, sa_idx) - 1]
            for sa_idx in [
                parrot.sa[i]
                for i in range(*parrot._range(kw.encode("utf-8")))
                if parrot.bwt_input_bytes[parrot.sa[i]:parrot.sa[i] + len(kw)] == kw.encode("utf-8")
            ]
        }
        matched_sets.append(matched_offsets)

    common_offsets = set.intersection(*matched_sets)
    return [parrot.paragraph_map[offset] for offset in sorted(common_offsets)]


def highlight_keywords(text, keywords, color="\033[1;31m"):
    """
    Highlights all keywords in text with ANSI color. Default = red.
    """
    def replacer(match):
        return f"{color}{match.group(0)}\033[0m"

    pattern = "|".join(re.escape(kw) for kw in keywords)
    return re.sub(f"(?i)({pattern})", replacer, text)

def extract_sentence_with_keyword(text, keyword):
    """
    Find sentence containing the keyword.
    """
    for sentence in re.split(r'(?<=[.!?])\s+', text):
        if keyword in sentence:
            return sentence
    return text[:200]

### Parrot Marokv

class ParrotMarkovPhrase:
    def __init__(self, model_file: str = "data/markov_model.pkl"):
        """
        Initializes the ParrotMarkov instance by loading a prebuilt
        phrase-level Markov chain and sentence bank.

        Parameters:
            model_file (str): Path to the pickle file containing
                              (markov_chain, sentence_bank)
        """
        with open(model_file, "rb") as f:
            self.chain, self.sentence_bank = pickle.load(f)

        self.order = len(next(iter(self.chain)))  # infer order from first key

### generate a phrase text based on the ParrotMarkovPhrase object and a prompt
### here, for comparison, the prompt is in the form of keyword list
def generate_phrase_text(parrot, prompt=None, order=2, length=6):
    chain = parrot.chain
    sentence_bank = parrot.sentence_bank 

    keys = list(chain.keys())
    start = None

    if prompt:
        prompt = [w.lower() for w in prompt]
        # Search for a phrase containing any of the keywords
        matching_keys = [
            key for key in keys if any(any(word in s.lower() for word in prompt) for s in key)
        ]
        if matching_keys:
            start = random.choice(matching_keys)

    if not start:
        start = random.choice(keys)

    output = list(start)

    for _ in range(length):
        key = tuple(output[-order:])
        next_phrases = chain.get(key)
        if not next_phrases:
            break
        next_phrase = random.choice(next_phrases)
        output.append(next_phrase)

    return " ".join(output)


# this code builds "indices" which facilitates search

import os
import re
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import pickle
import numpy as np
import pydivsufsort

QUESTION_WORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how"
}
STOPWORDS = set(stopwords.words('english')) | QUESTION_WORDS
VALID_POS = {"NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ", "JJR", "JJS", "RB", "RBR", "RBS"}

def extract_keywords(text):
    keywords = []

    # 1. Extract all standalone 4-digit numbers (e.g., years)
    years = re.findall(r'\b\d{4}\b', text)
    keywords.extend(years)

    # 2. Preserve capitalized multi-word phrases (e.g., “Mass in B Minor”)
    capitalized_phrases = re.findall(r'([A-Z][a-z]*(?:\s+[A-Z][a-z0-9]*){0,5})', text)
    keywords.extend(phrase.strip().lower() for phrase in capitalized_phrases if len(phrase.split()) <= 6)

    # 3. Tokenize and POS-tag
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    for word, tag in tagged:
        word_clean = re.sub(r'[^\w\-]', '', word.lower())

        if (
            tag in VALID_POS or
            re.fullmatch(r'\d+', word_clean)  # catch any remaining numbers
        ):
            if word_clean not in STOPWORDS and len(word_clean) > 1:
                keywords.append(word_clean)

    all_keywords = list(set(keywords))

    return all_keywords, " ".join(all_keywords)


def build_bwt_index_data(corpus_dir):
    bwt_input_str = ""
    paragraph_map = {}
    paragraph_offsets = []
    keyword_set = []

    index = 0  # Running offset in bwt_input_str

    for filename in os.listdir(corpus_dir):
        if not filename.endswith(".txt"):
            continue

        article = filename.replace(".txt", "")
        section = None

        with open(os.path.join(corpus_dir, filename), "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        for line in lines:
            if line.startswith("##"):
                section = line.strip("## ").strip()
                continue

            if len(line.strip()) < 40:
                continue

            full_text = line.strip()
            keywords, filtered_text = extract_keywords(full_text)

            if not filtered_text:
                continue

            # Track offset before appending
            paragraph_offsets.append(index)
            paragraph_map[index] = {
                "article": article,
                "section": section,
                "text": full_text,
                "keywords": keywords
            }

            # Append to BWT input string with separator
            bwt_input_str += filtered_text + "\x03"
            index = len(bwt_input_str)  # Update offset
            keyword_set.extend(keywords)

    return bwt_input_str, paragraph_offsets, paragraph_map, keyword_set

def build_suffix_array(text):
    return sorted(range(len(text)), key=lambda i: text[i:])
    
bwt_input_str, paragraph_offsets, paragraph_map, keyword_set = build_bwt_index_data('./data/corpus_raw/')

output_dir = "./data/bwt_index"  # Or wherever you store derived files
os.makedirs(output_dir, exist_ok=True)

# Save bwt_input_str
#with open(os.path.join(output_dir, "bwt_input_str.txt"), "w", encoding="utf-8") as f:
#    f.write(bwt_input_str)

# Save paragraph_offsets
with open(os.path.join(output_dir, "paragraph_offsets.pkl"), "wb") as f:
    pickle.dump(paragraph_offsets, f)

# Save paragraph_map
with open(os.path.join(output_dir, "paragraph_map.pkl"), "wb") as f:
    pickle.dump(paragraph_map, f)

# create indices for bwt_input_str

# Convert string to bytes and then to numpy array
#bwt_bytes = np.frombuffer(bwt_input_str.encode('utf-8'), dtype=np.uint8).copy()
bwt_input_bytes = bwt_input_str.encode("utf-8") 

print("Before SA build:")
print("len(bwt_input_bytes):", len(bwt_input_bytes))
print("Preview last 10 bytes:", bwt_input_bytes[-10:])


# save bwt_input_bytes
with open(os.path.join(output_dir, "bwt_input_bytes.pkl"), "wb") as f:
    pickle.dump(bwt_input_bytes, f)

# Generate suffix array
suffix_array = pydivsufsort.divsufsort(bwt_input_bytes)

print("Suffix array built")
print("Max index in SA:", max(suffix_array))

# Save index file
with open(os.path.join(output_dir, "sa.pkl"), "wb") as f:
    pickle.dump(suffix_array, f)

# Save keyword set
with open(os.path.join(output_dir,"keyword_set.pkl"), "wb") as f:
    pickle.dump(keyword_set, f)



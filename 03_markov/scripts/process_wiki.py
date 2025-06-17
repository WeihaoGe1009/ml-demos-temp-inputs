# this code converts wiki page to texts, maintaining headers

from bs4 import BeautifulSoup
import os
import re

def extract_paragraphs_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        html = f.read()

    soup = BeautifulSoup(html, "html.parser")

    tags = soup.find_all(['p', 'h2', 'h3'])

    paragraphs = []
    for tag in tags:
        # Remove superscript citations and tooltips
        for sup in tag.find_all(['sup', 'span']):
            sup.decompose()

        # Get heading or paragraph text
        text = tag.get_text(separator=' ', strip=True)

        # Remove extra brackets and [a], [1], etc.
        text = re.sub(r'\[[^\]]*\]', '', text)        # Remove [1], [a], etc.
        text = re.sub(r'\s+', ' ', text)              # Normalize whitespace
        text = re.sub(r'\s+([.,;:?!])', r'\1', text)   # Remove space before punctuation

        if tag.name in ['h2', 'h3']:
            text = f"## {text}"

        if len(text) > 50 and not text.lower().startswith(('see also', 'references', 'external links')):
            paragraphs.append(text)

    print(f"[✓] Extracted {len(paragraphs)} from {filepath}")
    return paragraphs

SAVE_DIR = "../data/corpus_raw"
os.makedirs(SAVE_DIR, exist_ok=True)

html_files = {
    "bach": "../data/raw_html/bach.html",
    "mozart": "../data/raw_html/mozart.html",
    "beethoven": "../data/raw_html/beethoven.html",
    "haydn": "../data/raw_html/haydn.html"
}

for name, path in html_files.items():
    paras = extract_paragraphs_from_file(path)
    with open(f"{SAVE_DIR}/{name}.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(paras))


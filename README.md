# Module 3 - Markov Chain: Language, Probability & Illusion

This module explores how machines can **generate text, music, and plausible responses** using simple probabilities — without understanding any meaning. Through the lens of **Markov chains**, we demonstrate how models can assemble sequences letter by letter, note by note, or word by word, guided only by the **likelihood of transitions**. What emerges may resemble real language or music, but is in fact built from chance — offering a glimpse into the **illusion of understanding** that statistical models can create.

---

## What’s Inside
- `03_markov.ipynb` - The interactive Google Colab notebook ([Run on Google Colab](https://colab.research.google.com/github/WeihaoGe1009/ml-demos-temp-inputs/blob/main/03_markov/03_markov.ipynb))
- `data/` - Contains data for section 2. Including: 
    - `html_raw/`: the raw wikipedia htmls 
    - `corpus_raw/`: the text files processed for model training
    - `bwt_index/`: the index file generated from the text files for searching
    - `markov_model.pkl`: a pretrained markov chain model  
- `scripts/` - Contains python scripts for section 2. Including:
    - `process_wiki.py`: preprocess raw html files and generate files in `corpus_raw`
    - `build_bwt_index.py`: process files in `corpus_raw` to generate index files
    - `build_markov.py`: pre-train the markov model and save it in `markov_model.pkl`
    - `parrots.py`: implementing the parrots and helper functions  

---

## What You'll Learn
- What is Markov chain?
- What is the difference between search and data generation, i.e. the difference between pre-AI era search engines and AI chatbots nowadays? 
- How does a generative model create music (an example of non-text work)? 

## Take-Home Messages
- Markov chain and other generative model produce sequences of data based on the probabilities. "There’s no understanding — just a chain of guesses."
- Search vs Generation: "Search" only looks for existing content. "Generation" produces content based on probability, and thus creates illusions.
- A Markov chain model generates new sequences — such as text, sounds, or images — by selecting each element based on probabilities. These probabilities are not invented by the machine; they are learned from observing patterns in existing human-created data during training.

## Data
- section 1: Brown Corpus from NLTK  
- section 2: Wikipedia pages of Bach, Haydn, Mozart, and Beethoven
- section 3: Synthetic note sequences generated via Markov chain; rendered as audio using FluidSynth


## Data Source:

- section 1 Brown Corpus from NLTK, public domain data 
- section 2, manually downloaded the following pages from Wikipedia
    - [Johann Sebastian Bach](https://en.wikipedia.org/wiki/Johann_Sebastian_Bach)
    - [Joseph Haydn](https://en.wikipedia.org/wiki/Joseph_Haydn)
    - [Wolfgang Amadeus Mozart](https://en.wikipedia.org/wiki/Wolfgang_Amadeus_Mozart)
    - [Ludwig van Beethoven](https://en.wikipedia.org/wiki/Ludwig_van_Beethoven)  
- section 3, Instrument sounds rendered using FluidSynth (LGPL-licensed open-source synthesizer with General MIDI soundfonts); note sequences fully synthetic  


## References
- **Markov Chain**: Meyn, Sean, and Richard L. Tweedie. Markov Chains and Stochastic Stability. 2nd ed., with a prologue by Peter W. Glynn, Cambridge University Press, 2009. 
- **Brown Corpus**: Francis, W. Nelson, and Henry Kucera. Brown Corpus. Compiled 1961. Accessed via the NLTK Toolkit, https://www.nltk.org/nltk\_data/. 
- **nltk**: Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O'Reilly Media Inc. 
- **pyFluidSynth**: PyFluidSynth Developers. pyFluidSynth: Python Bindings for the FluidSynth SoundFont Synthesizer. Python Package Index, https://pypi.org/project/pyFluidSynth/.
- **FluidSynth**: FluidSynth Developers. FluidSynth: A Software Synthesizer Based on the SoundFont 2 Specifications. https://www.fluidsynth.org/.

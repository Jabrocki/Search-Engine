import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter


class TextPreprocessor:
    def __init__(self):
        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = SnowballStemmer("english")

    def preprocess_text(self, text: str) -> Counter[str]:
        text = re.sub(r"[^a-z0-9\s]", "", text.lower())
        words = [w for w in text.split() if w not in self.stop_words]
        stemmed_words = [self.stemmer.stem(w) for w in words]

        all_tokens = list(stemmed_words)

        for n in range(2, 4):
            grams = ["_".join(g) for g in nltk.ngrams(stemmed_words, n)]
            all_tokens.extend(grams)

        return Counter(all_tokens)

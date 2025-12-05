import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text: str):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in STOPWORDS]
    return " ".join(words)

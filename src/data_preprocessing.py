import pandas as pd
import re
import yaml
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from logger import logger

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text, params):
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()

    words = text.split()

    if params["preprocessing"]["remove_stopwords"]:
        words = [w for w in words if w not in STOPWORDS]

    if params["preprocessing"]["stem"]:
        words = [stemmer.stem(w) for w in words]

    return " ".join(words)

def preprocess_data():
    df = pd.read_csv("data/raw.csv")
    params = yaml.safe_load(open("config/params.yaml"))

    logger.info("Cleaning text...")
    df["clean_text"] = df["verified_reviews"].apply(lambda x: clean_text(x, params))
    df["clean_text"].replace("", "unknown", inplace=True)
    df["length"] = df["verified_reviews"].apply(len)

    df.to_csv("data/processed.csv", index=False)
    logger.info("Saved processed data → data/processed.csv")

    return df

if __name__ == "__main__":
    preprocess_data()

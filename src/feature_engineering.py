import pickle
import yaml
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from logger import logger

def generate_features():
    params = yaml.safe_load(open("config/params.yaml"))
    df = pd.read_csv("data/processed.csv")

    logger.info("Building CountVectorizer...")
    cv = CountVectorizer(max_features=params["features"]["max_features"])
    X = cv.fit_transform(df["clean_text"]).toarray()
    y = df["feedback"].values

    pickle.dump(cv, open("models/countVectorizer.pkl", "wb"))
    logger.info("Saved CountVectorizer")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    pickle.dump(scaler, open("models/scaler.pkl", "wb"))
    logger.info("Saved MinMaxScaler")

    return X_scaled, y

if __name__ == "__main__":
    generate_features()

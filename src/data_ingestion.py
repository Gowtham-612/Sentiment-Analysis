import pandas as pd
import yaml
from logger import logger

def ingest_data():
    params = yaml.safe_load(open("config/params.yaml"))
    data_path = params["data_path"]

    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path, delimiter="\t", quoting=3)

    df.dropna(inplace=True)
    logger.info(f"Load dataset shape: {df.shape}")

    df.to_csv("data/raw.csv", index=False)
    logger.info("Saved raw data → data/raw.csv")

    return df

if __name__ == "__main__":
    ingest_data()

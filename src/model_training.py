import pickle
import yaml
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from logger import logger
from feature_engineering import generate_features

def train_model():
    params = yaml.safe_load(open("config/params.yaml"))

    X, y = generate_features()

    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["train_test_split"]["test_size"],
        random_state=params["train_test_split"]["random_state"]
    )

    mlflow.set_experiment("sentiment-analysis")

    with mlflow.start_run():
        model_params = params["model"]["xgb"]

        logger.info("Training XGBoost model...")
        model = XGBClassifier(
            n_estimators=model_params["n_estimators"],
            learning_rate=model_params["learning_rate"],
            max_depth=model_params["max_depth"],
            subsample=model_params["subsample"]
        )

        model.fit(X_train, y_train)

        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        mlflow.log_params(model_params)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy", test_acc)

        logger.info(f"Train Acc: {train_acc}, Test Acc: {test_acc}")

        mlflow.sklearn.log_model(model, name="model")

        pickle.dump(model, open("models/model_xgb.pkl", "wb"))
        logger.info("Saved model → models/model_xgb.pkl")

    return model, X_test, y_test

if __name__ == "__main__":
    train_model()

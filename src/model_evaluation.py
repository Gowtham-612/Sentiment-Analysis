import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from logger import logger
from model_training import train_model

def evaluate_model():
    model, X_test, y_test = train_model()

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    logger.info(f"Evaluation Accuracy: {acc}")

    cm = confusion_matrix(y_test, preds)
    logger.info(f"Confusion Matrix:\n{cm}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

if __name__ == "__main__":
    evaluate_model()

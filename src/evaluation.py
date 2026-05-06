import sklearn 
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    import os
    os.makedirs("outputs/results", exist_ok=True)

    with open("outputs/results/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}\n")
        f.write(f"Confusion Matrix:\n{cm}")

    pd.DataFrame(y_pred, columns=["predictions"]).to_csv(
        "outputs/results/predictions.csv", index=False
    ) 
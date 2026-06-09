import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# dagshub.init(repo_owner='razzan10', repo_name='MSML-Loan-Prediction', mlflow=True)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "https://dagshub.com/razzan10/MSML-Loan-Prediction.mlflow")
mlflow.set_tracking_uri(tracking_uri)

def main():
  mlflow.set_experiment("Loan_Prediction_Training")

  with mlflow.start_run():
    data_path = 'dataset_preprocessing/loan_data_clean.csv'
    df = pd.read_csv(data_path)

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    n_estimators = 100
    max_depth = 5
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    mlflow.autolog()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recal", rec)

    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Training selesai. accuracy: {acc:.4}")
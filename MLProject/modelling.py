import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

dagshub.init(repo_owner="razzan10", repo_name="MLFlow", mlflow=True)

df = pd.read_csv('dataset_preprocessing/loan_data_clean.csv')
X = df.drop('Loan_status', axis=1)
y = df['Loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("Loan_Prediction_Tuning")

with mlflow.start_run():
  param_grid = {'n_estimator': [50, 100], 'max_depth': [None, 5, 10]}
  rf = RandomForestClassifier(random_state=42)
  grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3)
  grid_search.fit(X_train, y_train)

  best_model = grid_search.best_estimator_
  precictions = best_model.predict(X_test)
  acc = accuracy_score(y_test, precictions)

  mlflow.log_params(grid_search.best_params_)
  mlflow.log_metric("accuracy", acc)

  mlflow.sklearn.log_model(best_model, "model")

  cm = confusion_matrix(y_test, predictions)
  plt.figure(figsize=(6,4))
  sns.heatmap(cm, annot=True, fmt='d')
  plt.savefig("confusion_matrix.png")
  mlflow.log_artifact("confusion_matrix.png")

  import json
  with open("metric_info.json", "w") as f:
    json.dump({"accuracy_score": acc}, f)
  mlflow.log_artifact("metric_info.json")

print(f"Train selesai. Akurasi:{acc}")

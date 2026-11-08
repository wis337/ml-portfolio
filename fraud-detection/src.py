import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, confusion_matrix, recall_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

import json
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"


def eda(df) -> None:
    # print(df.shape)
    print(df.head(5))
    df.info()
    print(df.describe())
    # print(df.isna().sum())
    print(df["Class"].value_counts(normalize=True))

    print(df.groupby('Class')['Amount'].describe(percentiles=[0.25, 0.5, 0.75])) 


def build_preprocessor(numeric_features):
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])
    transformers = [
        ("num", numeric_transformer, numeric_features)
    ]
    return ColumnTransformer(transformers=transformers)


def plot_threshold_metrics(y_true, proba):
    thresholds = np.linspace(0.01, 0.99, 100)
    precisions = []
    recalls = []
    
    false_positives = []
    false_negatives = []
    
    costs = []
    
    for t in thresholds:
        predictions = (proba >= t).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true,predictions).ravel()

        false_positives.append(fp)
        false_negatives.append(fn)
        
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)

        precisions.append(precision)
        recalls.append(recall)
        
        cost = fn * 200 + fp * 2
        costs.append(cost)
        
    
    plt.figure(figsize=(10, 5))

    plt.plot(thresholds, costs, label="Total Cost")

    plt.xlabel("Threshold")
    plt.ylabel("Cost")
    plt.title("Стоимость ошибок в зависимости от Threshold")

    plt.legend()
    plt.grid()
    plt.show()


def find_best_threshold(y_true, proba, cost_fn=100, cost_fp=2):
    thresholds = np.logspace(-3, 0, 100)
    
    costs = []
    
    for t in thresholds:
        predictions = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true,predictions).ravel()
        cost = fn * cost_fn + fp * cost_fp
        costs.append(cost)

    min_cost = min(costs)
    best_index = costs.index(min_cost)
    best_threshold = thresholds[best_index]

    print("Лучший threshold:", best_threshold)
    print("Минимальная стоимость:", min_cost)

    return best_threshold

def save_artifacts(pipe, threshold, cost_fn, cost_fp, features):
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "model.joblib")
    config = {
        "threshold": float(threshold),
        "cost_fn": cost_fn,
        "cost_fp": cost_fp,
        "features": features,
    }
    with open(MODELS_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def main() -> None:
    df = pd.read_csv(Path(__file__).parent / "creditcard.csv")   
    # eda(df)

    X = df.drop(columns=["Class", "Time"])
    y = df["Class"]
    numeric_features = df.drop(columns=["Class", "Time"]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def make_pipe(clf):
        preprocessor = build_preprocessor(numeric_features)
        return Pipeline([
            ("preprocessor", preprocessor), 
            ("classifier", clf)
        ])
    
    
    def threshold_study():
            cost_fn = 500
            cost_fp = 2
            pipe = make_pipe(lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.01, subsample_freq=1, subsample=0.8, colsample_bytree=0.8, reg_lambda=0, num_leaves=28, random_state=42, n_jobs = -1))
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            proba_cv = cross_val_predict(pipe, X_train, y_train, cv=skf, method="predict_proba")[:, 1]

            
            threshold = find_best_threshold(y_train, proba_cv, cost_fn=cost_fn, cost_fp=cost_fp)
            
            
            # Обучаем модель на всей обучающей выборке
            pipe.fit(X_train, y_train)

            # Получаем вероятности на тестовой выборке
            proba_test = pipe.predict_proba(X_test)[:, 1]
            pred_default = (proba_test >= 0.5).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_test, pred_default
            ).ravel()

            precision = precision_score(y_test, pred_default)
            recall = recall_score(y_test, pred_default)
            cost = fn * cost_fn + cost_fp * 2

            print("Результаты при threshold = 0.5")
            print("TP:", tp, "FP:", fp)
            print("TN:", tn, "FN:", fn)
            print("Precision:", precision)
            print("Recall:", recall)
            print("Cost:", cost)
            
            avg_prec = average_precision_score(y_test, proba_test)
            # Применяем найденный порог
            pred_test = (proba_test >= threshold).astype(int)
            # Считаем ошибки
            tn, fp, fn, tp = confusion_matrix(
                y_test,
                pred_test
            ).ravel()

            # Метрики
            precision = precision_score(y_test, pred_test, zero_division=0)
            recall = recall_score(y_test, pred_test, zero_division=0)

            # Стоимость ошибок
            cost = fn * cost_fn + fp * cost_fp

            print("\nРезультаты на TEST:")
            print(f"TP: {tp}, FP: {fp}")
            print(f"TN: {tn}, FN: {fn}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"Cost: {cost}")
            print("AVG_Prec: ", avg_prec)
            save_artifacts(pipe, threshold, cost_fn, cost_fp, numeric_features)
            
            
            
    def evaluate(name, clf):
        pipe = make_pipe(clf)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)        
        cv = cross_validate(pipe, X_train, y_train, cv = skf, scoring=["average_precision", "recall", "precision"], n_jobs=-1)
        print(f"{name}: avg_prec = {cv['test_average_precision'].mean():.3f} +/- {cv['test_average_precision'].std():.3f} | "
            f"recall={cv['test_recall'].mean():.3f} +/- std = {cv['test_recall'].std():.3f}  | "
            f"precision={cv['test_precision'].mean():.3f} +/- std = {cv['test_precision'].std():.3f}")
    # evaluate("LogReg", LogisticRegression(max_iter=1000))
    # evaluate("LogReg Balanced", LogisticRegression(max_iter=1000, class_weight="balanced"))
    # evaluate("Dummy", DummyClassifier(strategy="stratified"))   
    evaluate("LGBM", lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.01, subsample_freq=1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1, num_leaves=28, random_state=42, n_jobs = -1))
    # evaluate("RF", RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=10, n_jobs=-1, random_state=42))

    threshold_study()
    
    
if __name__ == "__main__":
    main()
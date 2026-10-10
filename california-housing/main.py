"""California Housing: регрессия цены жилья.

Пайплайн: EDA -> train/test -> baseline (LinearRegression) -> RandomForest
-> диагностика переобучения -> кросс-валидация -> GridSearchCV.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

RANDOM_STATE = 42


def eda(df: pd.DataFrame) -> None:
    print(df.shape)
    df.info()
    print(df.describe())
    print(df.isna().sum())

    plt.figure(figsize=(8, 5))
    df["MedHouseVal"].hist(bins=50)
    plt.title("Распределение MedHouseVal (обрезано на 5.0)")
    plt.xlabel("Цена, 100 000 $")
    plt.tight_layout()
    plt.show()

    corr = df.corr()["MedHouseVal"].drop("MedHouseVal").sort_values()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=corr.values, y=corr.index, hue=corr.index,
                palette="coolwarm", legend=False)
    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.title("Корреляция признаков с MedHouseVal")
    plt.xlabel("Коэффициент корреляции Пирсона")
    plt.tight_layout()
    plt.show()


def report(name, model, X_train, X_test, y_train, y_test) -> None:
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    print(f"{name}: "
          f"MAE train={mean_absolute_error(y_train, pred_train):.3f} "
          f"test={mean_absolute_error(y_test, pred_test):.3f} | "
          f"RMSE test={root_mean_squared_error(y_test, pred_test):.3f} | "
          f"R2 test={r2_score(y_test, pred_test):.3f}")


def main() -> None:
    df = fetch_california_housing(as_frame=True).frame
    eda(df)

    X = df.drop(columns="MedHouseVal")
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Baseline
    lr = LinearRegression().fit(X_train, y_train)
    report("LinearRegression", lr, X_train, X_test, y_train, y_test)

    # Лес с ручными параметрами
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=10,
        random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_train, y_train)
    report("RandomForest (md=15, msl=10)", rf, X_train, X_test, y_train, y_test)

    # Кросс-валидация
    for name, model in [("LinearRegression", LinearRegression()), ("RandomForest", rf)]:
        scores = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
        print(f"CV MAE {name}: {scores.mean():.3f} ± {scores.std():.3f}")

    # Подбор гиперпараметров (только на train)
    param_grid = {
        "max_depth": [5, 10, 15, 20],
        "min_samples_leaf": [1, 5, 10, 15],
    }
    grid = GridSearchCV(
        RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        param_grid, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print("Лучшие параметры:", grid.best_params_)
    print("Лучший CV MAE:", -grid.best_score_)

    # Обязательная проверка на переобучение
    report("RandomForest (GridSearchCV)", grid.best_estimator_,
           X_train, X_test, y_train, y_test)

    # Важность признаков
    importances = pd.Series(grid.best_estimator_.feature_importances_,
                            index=X.columns).sort_values()
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh")
    plt.title("Важность признаков (Random Forest)")
    plt.xlabel("Важность")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

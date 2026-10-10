"""Titanic: классификация выживших.

Весь препроцессинг (импутация, OneHot) живёт внутри Pipeline,
поэтому кросс-валидация и подбор параметров не содержат утечки данных.
"""
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

RANDOM_STATE = 42


def child_flag(a):
    """Флаг «ребёнок» считается после заполнения пропусков возраста."""
    return (a < 16).astype(int)


def eda(df) -> None:
    df.info()
    print(df.isna().sum())
    print(df["survived"].value_counts(normalize=True))

    plt.figure()
    sns.barplot(x="pclass", y="survived", hue="sex", data=df)
    plt.title("Выживаемость по классу и полу")
    plt.show()


def build_preprocessor(numeric_features, categorical_features, use_child_flag=True):
    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = [
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
    if use_child_flag:
        child_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("flag", FunctionTransformer(child_flag)),
        ])
        transformers.append(("child", child_transformer, ["age"]))
    return ColumnTransformer(transformers=transformers)


def main() -> None:
    df = sns.load_dataset("titanic")
    eda(df)

    # Признаки, считаемые построчно, можно создавать до split
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["fare_per_person"] = df["fare"] / df["family_size"]

    # Дубли (class, embark_town, who, adult_male) и утечка (alive) не берём
    features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked",
                "family_size", "fare_per_person"]
    numeric_features = ["pclass", "age", "sibsp", "parch", "fare",
                        "family_size", "fare_per_person"]
    categorical_features = ["sex", "embarked"]

    X = df[features]
    y = df["survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    def make_pipe(clf):
        return Pipeline([("preprocessor", preprocessor), ("classifier", clf)])

    def evaluate(name, clf):
        pipe = make_pipe(clf)
        pipe.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, pipe.predict(X_train))
        test_acc = accuracy_score(y_test, pipe.predict(X_test))
        acc = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
        auc = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
        print(f"{name}: train={train_acc:.3f} test={test_acc:.3f} | "
              f"CV acc={acc.mean():.3f}±{acc.std():.3f} | "
              f"CV AUC={auc.mean():.3f}±{auc.std():.3f}")

    evaluate("LogReg", LogisticRegression(max_iter=1000))
    evaluate("RF default", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    evaluate("RF limited", RandomForestClassifier(
        n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=RANDOM_STATE))
    evaluate("HistGB default", HistGradientBoostingClassifier(random_state=RANDOM_STATE))
    evaluate("HistGB limited", HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20,
        random_state=RANDOM_STATE))

    # Подробные метрики для baseline
    baseline = make_pipe(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    pred = baseline.predict(X_test)
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))
    proba = baseline.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", roc_auc_score(y_test, proba))


if __name__ == "__main__":
    main()

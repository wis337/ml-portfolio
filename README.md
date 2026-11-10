# ML Portfolio

Учебные проекты по машинному обучению на табличных данных. От первичного анализа и baseline до подбора порога по стоимости ошибок и сервиса с API.

| Проект                                    | Тип                                      | Что внутри                                                                                                                                                                       |
| ----------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [fraud-detection](fraud-detection/)       | Классификация, сильный дисбаланс классов | Baseline, логистическая регрессия, Random Forest, LightGBM, StratifiedKFold, Average Precision, подбор порога по стоимости ошибок, FastAPI-сервис. На тесте AP 0,88, recall 0,89 |
| [california-housing](california-housing/) | Регрессия                                | EDA, baseline, Random Forest, кросс-валидация, GridSearchCV, диагностика переобучения                                                                                            |
| [titanic](titanic/)                       | Классификация                            | Pipeline + ColumnTransformer, OneHotEncoder, метрики классификации, Random Forest и градиентный бустинг, feature engineering                                                     |

## Стек

Python, pandas, NumPy, scikit-learn, LightGBM, matplotlib, seaborn, FastAPI.

## Запуск

Данные больших проектов в репозитории не хранятся, ссылки на них есть в README соответствующих папок.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python california-housing/main.py
python titanic/main.py
```

У `fraud-detection` отдельные зависимости и свой запуск (обучение модели и API), подробности в [fraud-detection/README.md](fraud-detection/README.md):

```
cd fraud-detection
pip install -r requirements.txt
python src.py
uvicorn app:app --reload
```

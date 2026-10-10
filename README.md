# ML Portfolio

Учебные проекты по машинному обучению на табличных данных.

| Проект | Тип | Что внутри |
|---|---|---|
| [california-housing](california-housing/) | Регрессия | EDA, baseline, Random Forest, кросс-валидация, GridSearchCV, диагностика переобучения |
| [titanic](titanic/) | Классификация | Pipeline + ColumnTransformer, OneHotEncoder, метрики классификации, Random Forest и градиентный бустинг, feature engineering |

## Стек
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn.

## Запуск
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python california-housing/main.py
python titanic/main.py
```

from fastapi import FastAPI, HTTPException
import json
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
from pydantic import BaseModel
import pandas as pd


BASE_DIR = Path(__file__).parent / "models"
MODEL_PATH = BASE_DIR / "model.joblib"
CONFIG_PATH = BASE_DIR / "config.json"

#Контейнер для хранения глобальных ресурсов
ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    #Загрузка модели и конфига при старте приложения
    try:
        if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
            raise FileNotFoundError("Файлы модели или конфигурации не найдены в папке models/")
        
        ml_resources["model"] = joblib.load(MODEL_PATH)
        
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            ml_resources["config"] = json.load(f)

    except Exception as e:
        print(f"Ошибка при инициализации ML-компонентов: {e}")
        # Приложение не запустится, если критически важные файлы отсутствуют
        raise 
    
    yield
    # Очистка при выключении сервера
    ml_resources.clear()
    

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    features: dict[str, float]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest):
    model = ml_resources["model"]
    config = ml_resources["config"]
    
    expected_features = config["features"]
    threshold = config["threshold"]
    
    
    received = set(request.features)
    expected = set(expected_features)
    missing = sorted(expected - received)
    unexpected = sorted(received - expected)
    
    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": unexpected},
        )
    
    X = pd.DataFrame([request.features], columns=expected_features)
    
    score = float(model.predict_proba(X)[0, 1])
 
    return {
        "score": score,
        "is_fraud": score >= threshold,
        "applied_threshold": threshold,
    }

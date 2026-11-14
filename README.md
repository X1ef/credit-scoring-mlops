# Credit Default Prediction — MLOps

End-to-end учебный проект по предсказанию дефолта кредитной карты: валидация данных,
Feature Engineering, sklearn Pipeline, 5 MLflow экспериментов, DVC, pytest/CI,
FastAPI, Docker и мониторинг PSI.

## Быстрый старт

Python 3.11+, Git и (для контейнера) Docker.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
python -m credit_scoring.data --output data/raw/UCI_Credit_Card.csv
python -m credit_scoring.validation --data data/raw/UCI_Credit_Card.csv
python -m credit_scoring.prepare
# в отдельном терминале: mlflow ui --host 127.0.0.1 --port 5000
python -m credit_scoring.train
```

Если загрузка UCI недоступна, положите Kaggle CSV в `data/raw/UCI_Credit_Card.csv`.
Ожидается исходный датасет Default of Credit Card Clients и колонка
`default.payment.next.month`.

## DVC

```bash
dvc init
dvc add data/raw/UCI_Credit_Card.csv
git add .dvc .gitignore data/raw/UCI_Credit_Card.csv.dvc
git commit -m "Track raw dataset with DVC"
dvc repro
```

DVC pipeline содержит `prepare` и `train`. Для командной работы настройте remote:
`dvc remote add -d storage <remote-url>`.

## API и Docker

После обучения модель находится в `models/credit_default_model.joblib`.

```bash
uvicorn credit_scoring.api.app:app --host 0.0.0.0 --port 8000
docker compose up --build
```

Swagger: http://localhost:8000/docs; healthcheck: `/health`; инференс: `POST /predict`.
Пример JSON:
```json
{
  "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30,
  "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
  "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 13000,
  "BILL_AMT4": 12000, "BILL_AMT5": 11000, "BILL_AMT6": 10000,
  "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
  "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000
}
```

## Мониторинг

Запустите API и выполните:
`python -m credit_scoring.monitoring --data data/processed/test.csv --api-url http://localhost:8000`.
Скрипт отправляет test-наблюдения в API и рассчитывает PSI распределения PD
относительно train reference scores. PSI — сигнал изменения распределения, а не
самостоятельное доказательство ухудшения качества; для этого нужны новые метки.

## Что демонстрировать для максимальной оценки

1. GitHub: структура, README, осмысленные коммиты.
2. Pandera: успешная проверка исходных данных и тесты на аномалии.
3. EDA, агрегаты истории платежей, AGE_BIN, train/test и метрики ROC-AUC/Precision/
   Recall/F1 + ROC curve.
4. MLflow UI: минимум пять runs, параметры, метрики, ROC-графики, sklearn-модели.
5. `dvc dag` и успешный `dvc repro`.
6. GitHub Actions: Black, Flake8, pytest и валидация данных.
7. Docker build/run, Swagger и рабочий `/predict`.
8. PSI-отчёт.
9. Объяснение ограничений: датасет исторический, тайваньский и учебный; порог 0.5
   не является оптимальным бизнес-порогом. Для реального PD нужны временная
   валидация, калибровка, контроль утечки, независимая проверка и правовой анализ.

Метрики Precision/Recall/F1 считаются при threshold=0.5. Выбор рабочего порога
следует обосновывать экономикой ошибок и ограничениями кредитной политики.

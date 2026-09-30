FROM python:3.11-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends gcc git && rm -rf /var/lib/apt/lists/*

# copy project files
COPY pyproject.toml README.md /app/
COPY src /app/src

ENV PYTHONPATH=/app/src

# install runtime deps
RUN pip install --upgrade pip setuptools wheel
RUN pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

CMD ["python", "-m", "mobile_price_predictor.train"]

FROM python:3.11-slim

WORKDIR /app

# system deps for some packages (if needed)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app
RUN pip install -e .

EXPOSE 8501

CMD ["streamlit", "run", "src/mobile_price_predictor/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]

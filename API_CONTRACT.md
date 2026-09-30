# API Contract — Mobile Price Predictor

This document describes the HTTP API contract between the UI and backend.

Base URL

- Default (local): `http://localhost:8000`

Endpoints

1. GET /models
   - Description: list available model identifiers (filenames).
   - Response: 200 OK
     - Content-Type: application/json
     - Body: JSON array of strings, e.g.
       ```json
       ["predictor_v1_20260930.joblib", "predictor_v2_20261001.joblib"]
       ```

2. POST /predict
   - Description: run prediction on a single row of features.
   - Request: JSON
     - Content-Type: application/json
     - Body schema:
       ```json
       {
         "model": "optional-model-filename",
         "features": { "ram": 2000, "battery_power": 1500 }
       }
       ```
     - `model` (optional): if omitted, the server will use the latest available model.
     - `features`: mapping of column name -> value (numeric or string). Server will select numeric columns.

   - Response: 200 OK
     - Content-Type: application/json
     - Body:
       ```json
       { "predictions": [1] }
       ```

   - Errors:
     - 400 Bad Request — malformed JSON or missing `features`.
     - 404 Not Found — requested model not found or no models available.
     - 500 Internal Server Error — model prediction failed.

3. POST /reload
   - Description: reload models from disk.
   - Response: 200 OK { "loaded": <n> }

Notes and conventions

- The `features` object should include only the feature columns the model expects. The server will align, fill missing numeric columns with 0, and drop unexpected columns prior to prediction.
- Responses are plain JSON; no authentication is included by default. For production, add an auth layer (API keys, JWT, or reverse proxy).
- Keep model filenames stable; the UI uses filenames as model identifiers.

Examples

- Predict using `curl`:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": {"ram": 2000, "battery_power": 1500}}'
```

Security

- Do not expose the API publicly without adding authentication and HTTPS. Use a reverse proxy (nginx) or API gateway in production.

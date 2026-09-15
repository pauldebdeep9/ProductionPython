FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

WORKDIR /app

COPY pyproject.toml README.md ./
RUN python -m pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser 08_ai_service_capstone.py ./

USER appuser

EXPOSE 8000

CMD ["python", "08_ai_service_capstone.py", "--serve"]

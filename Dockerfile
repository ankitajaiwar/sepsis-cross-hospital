FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY artifacts/model/ artifacts/model/
ENV PYTHONPATH=/app/src MODEL_DIR=/app/artifacts/model
RUN useradd -m app
USER app
EXPOSE 8000
CMD ["uvicorn", "sepsis.api:app", "--host", "0.0.0.0", "--port", "8000"]

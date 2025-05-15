FROM python:3.11-slim

WORKDIR /app

# 환경 변수 설정 (버퍼링 방지 및 경로 설정)
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
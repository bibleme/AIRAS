# 1. Python 3.11 슬림 버전을 기반 이미지로 사용
FROM python:3.11-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 환경 변수 설정 (버퍼링 방지 및 경로 설정)
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app

# 4. 필요한 시스템 패키지 설치 (선택 사항, 필요 시 추가)
# 예: apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# 5. requirements.txt 파일을 먼저 복사하여 Docker 캐시 활용
COPY requirements.txt .

# 6. pip 업그레이드 및 의존성 설치 (no-cache-dir로 이미지 크기 최적화)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. NLTK 데이터 다운로드 (빌드 시점에 처리)
RUN python -m nltk.downloader punkt

# 8. 애플리케이션 소스 코드 복사
COPY . .

# 9. 애플리케이션이 사용할 포트 노출
EXPOSE 8000

# 10. 컨테이너 실행 시 uvicorn 서버 실행
# --host 0.0.0.0 옵션으로 외부에서 접근 가능하도록 설정
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .

# SQLite volume is mounted at /data; create the mount point so non-root users
# can use it if the host directory is chowned correctly.
RUN mkdir -p /data

EXPOSE 8000

CMD ["uvicorn", "dashboard.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY brain_trader/ brain_trader/

RUN pip install --no-cache-dir -e .

COPY .env.example .env

ENTRYPOINT ["brain-trader"]
CMD ["--help"]

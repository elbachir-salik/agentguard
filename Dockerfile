FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY agentguard/ agentguard/

RUN pip install --no-cache-dir -e ".[dashboard]"

ENV AGENTGUARD_DB_PATH=/data/agentguard.db
EXPOSE 8585

CMD ["agentguard", "dashboard", "--host", "0.0.0.0", "--port", "8585"]

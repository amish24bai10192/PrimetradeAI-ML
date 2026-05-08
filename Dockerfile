FROM python:3.9-slim

# Non-root user for security best practice
RUN useradd --create-home appuser
WORKDIR /app

# Install dependencies first (layer-caching friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source + data
COPY run.py config.yaml data.csv ./

# Switch to non-root
USER appuser

# Default command — no hard-coded paths; flags name every file explicitly
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]

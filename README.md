# MLOps Batch Job — Rolling-Mean Signal Pipeline

A minimal MLOps-style batch job that demonstrates **reproducibility**, **observability**, and **deployment readiness** via a Dockerised, one-command pipeline.

---

## What it does

| Step | Description |
|------|-------------|
| 1 | Load & validate `config.yaml` (seed, window, version) |
| 2 | Load & validate `data.csv` — checks missing file, bad CSV, empty file, missing `close` column |
| 3 | Compute rolling mean on `close` with configurable `window` |
| 4 | Generate binary signal: `1` if `close > rolling_mean`, else `0` |
| 5 | Write structured `metrics.json` + detailed `run.log` |

The first `window - 1` rows produce `NaN` rolling-mean values and are **excluded** from signal computation and metric counts — this behaviour is consistent and documented.

---

## Project structure

```
.
├── run.py           # Main pipeline
├── config.yaml      # Seed, window, version config
├── data.csv         # 10 000-row OHLCV dataset
├── requirements.txt # Python dependencies
├── Dockerfile       # Docker build spec
├── metrics.json     # Sample successful output
├── run.log          # Sample log output
└── README.md
```

---

## Local run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Command

```bash
python run.py \
  --input    data.csv \
  --config   config.yaml \
  --output   metrics.json \
  --log-file run.log
```

Final metrics JSON is printed to **stdout**; all structured logs go to `run.log`.

---

## Docker build & run

```bash
# Build
docker build -t mlops-task .

# Run
docker run --rm mlops-task
```

- Exit code `0` → success  
- Exit code non-zero → failure (error details in `metrics.json` and stdout)

### Copy outputs from container (optional)

```bash
docker run --rm -v "$(pwd)/out":/app/out mlops-task \
  python run.py \
    --input    data.csv \
    --config   config.yaml \
    --output   out/metrics.json \
    --log-file out/run.log
```

---

## Config reference (`config.yaml`)

| Key | Type | Description |
|-----|------|-------------|
| `seed` | int | NumPy random seed for reproducibility |
| `window` | int ≥ 1 | Rolling-mean window size |
| `version` | str | Pipeline version tag written to metrics |

---

## Example `metrics.json` (success)

```json
{
  "version": "v1",
  "rows_processed": 9996,
  "metric": "signal_rate",
  "value": 0.5118,
  "latency_ms": 41,
  "seed": 42,
  "status": "success"
}
```

> `rows_processed` = total rows minus the `window - 1` warm-up rows excluded from signal computation.

## Example `metrics.json` (error)

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Required column 'close' not found. Available columns: ['open', 'high']"
}
```

`metrics.json` is **always written** — even on failure — so downstream monitors always have a machine-readable status.

---

## Reproducibility

Running the pipeline multiple times with the same `config.yaml` and `data.csv` produces **identical** `value` and `rows_processed` outputs. The `latency_ms` field will naturally vary with hardware load.

---

## Validation errors handled

| Case | Behaviour |
|------|-----------|
| Missing input file | Error metrics written, exit 1 |
| Non-parseable CSV | Error metrics written, exit 1 |
| Empty CSV | Error metrics written, exit 1 |
| Missing `close` column | Error metrics written, exit 1 |
| Missing config keys | Error metrics written, exit 1 |
| Non-numeric `close` rows | Warning logged, rows dropped, pipeline continues |

"""
MLOps Batch Job — Rolling-mean signal pipeline.
Usage:
    python run.py --input data.csv --config config.yaml \
                  --output metrics.json --log-file run.log
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps rolling-mean signal pipeline")
    parser.add_argument("--input",    required=True, help="Path to input CSV (OHLCV)")
    parser.add_argument("--config",   required=True, help="Path to YAML config file")
    parser.add_argument("--output",   required=True, help="Path for output metrics JSON")
    parser.add_argument("--log-file", required=True, dest="log_file",
                        help="Path for structured log output")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S"
    )
    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Console handler (stderr so stdout stays clean for JSON)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# Config loading + validation
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping at the top level.")

    missing = REQUIRED_CONFIG_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"Config is missing required keys: {missing}")

    # Type checks
    if not isinstance(cfg["seed"], int):
        raise ValueError(f"'seed' must be an integer, got: {type(cfg['seed']).__name__}")
    if not isinstance(cfg["window"], int) or cfg["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got: {cfg['window']}")
    if not isinstance(cfg["version"], str) or not cfg["version"].strip():
        raise ValueError(f"'version' must be a non-empty string, got: {cfg['version']!r}")

    return cfg


# ---------------------------------------------------------------------------
# Dataset loading + validation
# ---------------------------------------------------------------------------

def load_dataset(input_path: str) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV is empty (no data rows).")

    if "close" not in df.columns:
        raise ValueError(
            f"Required column 'close' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Coerce close to numeric; non-parseable values become NaN
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    n_invalid = df["close"].isna().sum()
    if n_invalid == len(df):
        raise ValueError("Column 'close' contains no valid numeric values.")
    if n_invalid > 0:
        # Warn but continue — drop bad rows
        logging.getLogger("mlops").warning(
            "%d row(s) with non-numeric 'close' dropped.", n_invalid
        )
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def compute_rolling_mean(close: pd.Series, window: int) -> pd.Series:
    """
    Rolling mean with min_periods=window so the first (window-1) rows are NaN.
    Those rows are excluded from signal computation.
    """
    return close.rolling(window=window, min_periods=window).mean()


def compute_signal(close: pd.Series, rolling_mean: pd.Series) -> pd.Series:
    """
    signal = 1 if close > rolling_mean else 0.
    Rows where rolling_mean is NaN are excluded (signal = NaN → cast to Int64).
    """
    valid_mask = rolling_mean.notna()
    signal = pd.Series(np.nan, index=close.index)
    signal[valid_mask] = (close[valid_mask] > rolling_mean[valid_mask]).astype(int)
    return signal


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_metrics(output_path: str, payload: dict) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_error_metrics(output_path: str, version: str, message: str) -> None:
    payload = {
        "version": version,
        "status": "error",
        "error_message": message,
    }
    write_metrics(output_path, payload)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    logger = setup_logging(args.log_file)

    # Default version for error payloads before config is loaded
    version = "unknown"
    start_time = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Job started")
    logger.info("Input  : %s", args.input)
    logger.info("Config : %s", args.config)
    logger.info("Output : %s", args.output)
    logger.info("Log    : %s", args.log_file)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1 — Load + validate config
    # ------------------------------------------------------------------
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        logger.exception("Config validation failed: %s", exc)
        write_error_metrics(args.output, version, str(exc))
        return 1

    version = cfg["version"]
    seed    = cfg["seed"]
    window  = cfg["window"]

    logger.info("Config loaded  — version=%s | seed=%d | window=%d",
                version, seed, window)

    # Set global seed for reproducibility
    np.random.seed(seed)
    logger.info("NumPy random seed set to %d", seed)

    # ------------------------------------------------------------------
    # Step 2 — Load + validate dataset
    # ------------------------------------------------------------------
    try:
        df = load_dataset(args.input)
    except Exception as exc:
        logger.exception("Dataset loading failed: %s", exc)
        write_error_metrics(args.output, version, str(exc))
        return 1

    logger.info("Dataset loaded — %d rows, columns: %s",
                len(df), list(df.columns))

    # ------------------------------------------------------------------
    # Step 3 — Rolling mean
    # ------------------------------------------------------------------
    logger.info("Computing rolling mean (window=%d) on 'close' …", window)
    df["rolling_mean"] = compute_rolling_mean(df["close"], window)
    valid_rows = df["rolling_mean"].notna().sum()
    logger.info("Rolling mean computed — %d rows with valid mean "
                "(%d warm-up rows excluded from signal)",
                valid_rows, len(df) - valid_rows)

    # ------------------------------------------------------------------
    # Step 4 — Signal generation
    # ------------------------------------------------------------------
    logger.info("Generating binary signal (close > rolling_mean → 1, else 0) …")
    df["signal"] = compute_signal(df["close"], df["rolling_mean"])

    signal_valid = df["signal"].dropna()
    rows_processed = len(signal_valid)
    signal_rate    = float(signal_valid.mean())
    logger.info("Signal generated — rows_processed=%d | signal_rate=%.6f",
                rows_processed, signal_rate)

    # ------------------------------------------------------------------
    # Step 5 — Metrics + timing
    # ------------------------------------------------------------------
    elapsed_ms = round((time.perf_counter() - start_time) * 1000)

    metrics = {
        "version":        version,
        "rows_processed": rows_processed,
        "metric":         "signal_rate",
        "value":          round(signal_rate, 4),
        "latency_ms":     elapsed_ms,
        "seed":           seed,
        "status":         "success",
    }

    write_metrics(args.output, metrics)
    logger.info("Metrics written to %s", args.output)
    logger.info("Metrics summary — %s", json.dumps(metrics))

    logger.info("Job finished successfully | latency_ms=%d", elapsed_ms)
    logger.info("=" * 60)

    # Print metrics JSON to stdout (required by Docker spec)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

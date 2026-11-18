# Embedding Drift Monitor

An offline Python demo for monitoring a change in vector distributions. Compare embeddings from the same model and preprocessing pipeline in two batches, then inspect a normalized Jensen–Shannon divergence on their cosine similarity to the reference centroid.

```text
reference vectors ──> normalize ──> reference centroid ─┐
current vectors ────> normalize ────────────────────────┴─> cosine histograms → JS divergence → alert
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
embedding-drift --reference examples/reference.json --current examples/current.json --threshold 0.1
pytest -q
```

The examples are synthetic 2D vectors, not measurements from a deployed model. Inputs are JSON arrays of equal-length numeric vectors; each batch needs at least two finite nonzero vectors. Exit 0 means divergence is at or below the threshold, 1 means above, and 2 means invalid input. Output includes sample counts, dimension, average similarities and divergence.

## Interpretation and limits

This tool does not generate embeddings, make model calls or measure model quality. Compare only outputs from the same model version and preprocessing. The scalar similarity-to-centroid projection loses directional detail, so shifts can go undetected. Histograms depend on bin boundaries and sample size; calibrate thresholds against a representative baseline, not the synthetic example. It is a monitoring aid, not an automatic retraining decision or proof of production drift.

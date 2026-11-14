import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from credit_scoring.features import TARGET


def calculate_psi(reference, current, bins=10, epsilon=1e-6):
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a, _ = np.histogram(ref, bins=edges)
    b, _ = np.histogram(cur, bins=edges)
    p = np.clip(a / max(a.sum(), 1), epsilon, None)
    q = np.clip(b / max(b.sum(), 1), epsilon, None)
    return float(np.sum((q - p) * np.log(q / p)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/test.csv")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--sample-size", type=int, default=500)
    args = parser.parse_args()
    ref_path = Path("monitoring/reference_scores.npy")
    if not ref_path.exists():
        raise FileNotFoundError("Train model first; reference scores are missing")
    ref = np.load(ref_path)
    df = pd.read_csv(args.data)
    sample = df.sample(n=min(args.sample_size, len(df)), random_state=42)
    scores = []
    for row in sample.drop(columns=[TARGET], errors="ignore").to_dict("records"):
        response = requests.post(
            args.api_url.rstrip("/") + "/predict", json=row, timeout=30
        )
        response.raise_for_status()
        scores.append(response.json()["default_probability"])
    psi = calculate_psi(ref, scores)
    report = {
        "metric": "PSI",
        "population": "predicted_default_probability",
        "psi": psi,
        "reference_size": len(ref),
        "current_size": len(scores),
        "interpretation": (
            "stable"
            if psi < 0.1
            else "moderate_shift" if psi < 0.25 else "significant_shift"
        ),
    }
    Path("monitoring/psi_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

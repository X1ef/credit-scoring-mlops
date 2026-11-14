import argparse
from pathlib import Path
import pandas as pd


def download_uci_dataset(output):
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=350)
    x = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].rename("default.payment.next.month")
    result = pd.concat([x, y], axis=1)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw/UCI_Credit_Card.csv")
    args = parser.parse_args()
    print(f"Saved: {download_uci_dataset(args.output)}")


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from credit_scoring.features import TARGET, engineer_features
from credit_scoring.validation import validate_credit_data


def prepare_data(
    raw_path="data/raw/UCI_Credit_Card.csv",
    output="data/processed",
    test_size=0.2,
    random_state=42,
):
    raw = pd.read_csv(raw_path)
    if "ID" in raw.columns:
        raw = raw.drop(columns=["ID"])
    validate_credit_data(raw)
    data = engineer_features(raw)
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data[TARGET]
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    return train, test


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())["prepare"]
    prepare_data(test_size=params["test_size"], random_state=params["random_state"])


if __name__ == "__main__":
    main()

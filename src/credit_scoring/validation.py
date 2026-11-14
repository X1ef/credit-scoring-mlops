import argparse
import pandas as pd
import pandera.pandas as pa
from pandera import Check
from credit_scoring.features import BASE_COLS, TARGET

RANGES = {
    "LIMIT_BAL": (0, 10_000_000),
    "AGE": (18, 100),
    "SEX": (1, 2),
    "EDUCATION": (0, 6),
    "MARRIAGE": (0, 3),
    **{f"PAY_{i}": (-2, 9) for i in [0, 2, 3, 4, 5, 6]},
}


def credit_schema():
    columns = {c: pa.Column(float, nullable=False, coerce=True) for c in BASE_COLS}
    for name, (low, high) in RANGES.items():
        columns[name] = pa.Column(
            float, nullable=False, coerce=True, checks=Check.in_range(low, high)
        )
    columns[TARGET] = pa.Column(
        int, nullable=False, coerce=True, checks=Check.isin([0, 1])
    )
    return pa.DataFrameSchema(columns, strict=False, coerce=True)


def validate_credit_data(frame):
    missing = sorted(set(BASE_COLS + [TARGET]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return credit_schema().validate(frame, lazy=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    validate_credit_data(df)
    print(f"Validation passed: {len(df)} rows")


if __name__ == "__main__":
    main()

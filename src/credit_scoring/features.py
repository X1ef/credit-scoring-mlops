import pandas as pd

TARGET = "default.payment.next.month"
CAT_COLS = [
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
]
BASE_COLS = (
    [
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    bills = [f"BILL_AMT{i}" for i in range(1, 7)]
    payments = [f"PAY_AMT{i}" for i in range(1, 7)]
    statuses = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    data["BILL_AMT_MEAN"] = data[bills].mean(axis=1)
    data["BILL_AMT_MAX"] = data[bills].max(axis=1)
    data["PAY_AMT_MEAN"] = data[payments].mean(axis=1)
    data["PAYMENT_TO_BILL_RATIO"] = data[payments].sum(axis=1) / data[bills].abs().sum(
        axis=1
    ).replace(0, 1)
    data["DELAYED_MONTHS"] = (data[statuses] > 0).sum(axis=1)
    data["AGE_BIN"] = pd.cut(
        data["AGE"],
        [0, 25, 35, 45, 55, 120],
        labels=["18-25", "26-35", "36-45", "46-55", "56+"],
        include_lowest=True,
    ).astype("object")
    return data

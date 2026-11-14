import pandas as pd
from credit_scoring.features import engineer_features


def test_feature_engineering():
    row = {"LIMIT_BAL": 10000, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 30}
    row.update({f"PAY_{i}": 0 for i in [0, 2, 3, 4, 5, 6]})
    row.update({f"BILL_AMT{i}": 1000 for i in range(1, 7)})
    row.update({f"PAY_AMT{i}": 100 for i in range(1, 7)})
    out = engineer_features(pd.DataFrame([row]))
    assert out.loc[0, "BILL_AMT_MEAN"] == 1000
    assert out.loc[0, "DELAYED_MONTHS"] == 0
    assert "AGE_BIN" in out

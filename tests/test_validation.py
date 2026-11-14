from pathlib import Path
import pandas as pd
import pytest
from credit_scoring.validation import validate_credit_data

FIXTURE = Path(__file__).parent / "fixtures/valid_credit_sample.csv"


def test_fixture_validates():
    assert len(validate_credit_data(pd.read_csv(FIXTURE))) == 12


def test_invalid_target_fails():
    data = pd.read_csv(FIXTURE)
    data.loc[0, "default.payment.next.month"] = 8
    with pytest.raises(Exception):
        validate_credit_data(data)


def test_missing_column_fails():
    data = pd.read_csv(FIXTURE).drop(columns=["AGE"])
    with pytest.raises(ValueError):
        validate_credit_data(data)

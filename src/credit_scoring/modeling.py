from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from credit_scoring.features import CAT_COLS


def preprocessor(columns):
    categorical = [c for c in CAT_COLS + ["AGE_BIN"] if c in columns]
    numeric = [c for c in columns if c not in categorical]
    num_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)]
    )


def candidates(seed=42):
    return [
        ("logreg_c1", LogisticRegression(max_iter=1500, C=1.0)),
        ("logreg_c01", LogisticRegression(max_iter=1500, C=0.1)),
        (
            "rf_300",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "rf_depth8",
            RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        (
            "hist_gradient",
            HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.08,
                l2_regularization=1.0,
                random_state=seed,
            ),
        ),
    ]


def make_pipeline(model, columns):
    return Pipeline([("preprocess", preprocessor(columns)), ("model", model)])

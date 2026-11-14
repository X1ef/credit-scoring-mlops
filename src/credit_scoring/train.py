import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from credit_scoring.features import TARGET
from credit_scoring.metrics import calculate_metrics
from credit_scoring.modeling import candidates, make_pipeline


def train_experiments(
    train_path="data/processed/train.csv",
    test_path="data/processed/test.csv",
    random_state=42,
    cv=3,
    scoring="roc_auc",
):
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    x_train, y_train = train.drop(columns=[TARGET]), train[TARGET].astype(int)
    x_test, y_test = test.drop(columns=[TARGET]), test[TARGET].astype(int)
    mlflow.set_experiment("credit-default-prediction")
    best_auc, best_model, best_result, best_name = -1, None, None, None

    for name, estimator in candidates(random_state):
        pipe = make_pipeline(estimator, list(x_train.columns))
        with mlflow.start_run(run_name=name):
            cv_scores = cross_val_score(
                pipe, x_train, y_train, cv=cv, scoring=scoring, n_jobs=1
            )
            pipe.fit(x_train, y_train)
            probs = pipe.predict_proba(x_test)[:, 1]
            result = calculate_metrics(y_test, probs)
            mlflow.log_params(
                {"model": name, "cv": cv, "scoring": scoring, "threshold": 0.5}
            )
            mlflow.log_metric("cv_roc_auc_mean", float(cv_scores.mean()))
            mlflow.log_metric("cv_roc_auc_std", float(cv_scores.std()))
            mlflow.log_metrics(
                {k: v for k, v in result.items() if isinstance(v, (float, int))}
            )
            fpr, tpr, _ = roc_curve(y_test, probs)
            Path("reports/figures").mkdir(parents=True, exist_ok=True)
            fig = Path("reports/figures") / f"roc_{name}.png"
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={result['roc_auc']:.4f}")
            plt.plot([0, 1], [0, 1], "--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC — {name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig)
            plt.close()
            mlflow.log_artifact(str(fig))
            mlflow.sklearn.log_model(
                sk_model=pipe, name="model", serialization_format="cloudpickle"
            )
            if result["roc_auc"] > best_auc:
                best_auc, best_model, best_result, best_name = (
                    result["roc_auc"],
                    pipe,
                    result,
                    name,
                )

    Path("models").mkdir(exist_ok=True)
    joblib.dump(best_model, "models/credit_default_model.joblib")
    Path("monitoring").mkdir(exist_ok=True)
    np.save("monitoring/reference_scores.npy", best_model.predict_proba(x_train)[:, 1])
    Path("metrics").mkdir(exist_ok=True)
    output = {"best_model": best_name, **best_result}
    Path("metrics/metrics.json").write_text(json.dumps(output, indent=2))
    return output


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())["train"]
    print(
        json.dumps(
            train_experiments(
                random_state=params["random_state"],
                cv=params["cv"],
                scoring=params["scoring"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

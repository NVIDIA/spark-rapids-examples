from typing import List

import joblib
import optuna
import pandas as pd
import xgboost as xgb
from joblibspark import register_spark
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

register_spark()


def task(num_trials: int = 100):
    """
    This demo is to distribute xgboost training on spark cluster by referring to
    https://forecastegy.com/posts/xgboost-hyperparameter-tuning-with-optuna/
    """
    # url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    url = "/home/bobwang/data/xgboost/winequality-red.csv"
    data = pd.read_csv(url, delimiter=";")

    X = data.drop("quality", axis=1)
    y = data["quality"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # The objective function will be executed on the node for a total of "num_trials" times.
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "n_estimators": 1000,
            "verbosity": 0,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        predictions = model.predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=False)
        return rmse

    study = optuna.load_study(
        study_name="optuna-spark-xgboost", storage="mysql://optuna_user:optuna_password@10.19.129.248/optuna"
    )
    study.optimize(objective, n_trials=num_trials)
    return study.best_params, study.best_value


def partition_trials(total_trials: int, total_tasks: int) -> List[int]:
    # Calculate the base size of each partition
    base_size = total_trials // total_tasks
    # Calculate the number of partitions that will have an extra trial
    extra = total_trials % total_tasks

    # Create the partitions
    partitions = [base_size] * total_tasks
    for i in range(extra):
        partitions[i] += 1

    return partitions


# The total trials need to be run
total_trials = 100
# How many spark tasks will be launched, Please make sure total_tasks should be <= parallelism of spark
total_tasks = 2

with joblib.parallel_backend("spark", n_jobs=8):
    results = joblib.Parallel()(
        joblib.delayed(task)(i) for i in partition_trials(total_trials, total_tasks)
    )

best_params = min(results, key=lambda x: x[1])[0]
best_value = min(results, key=lambda x: x[1])[1]

print(f"Best parameters: {best_params}")
print(f"Best value: {best_value}")

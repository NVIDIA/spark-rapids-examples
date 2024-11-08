#
# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from typing import List
import joblib
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from joblibspark import register_spark
from pyspark import TaskContext
from sklearn.metrics import mean_squared_error as mean_squared_error
from sklearn.model_selection import train_test_split as train_test_split
import time

#optuna.logging.set_verbosity(optuna.logging.DEBUG)
register_spark()

# Get the databricks driver IP to connect to the MySQL server.
driver_ip = spark.conf.get("spark.driver.host", None)

if driver_ip is None:
    raise ValueError("Driver IP could not be retrieved from Spark configuration.")

def task(num_trials: int):
    """
    This demo is to benchmark distributed xgboost training on spark cluster.
    """

    start_etl = time.time()
  
    # read data from dbfs
    start_read = time.time()
    file_path = "/dbfs/FileStore/path/to/reg_2m_156_f32.parquet"
    data = pd.read_parquet(file_path)
    end_read = time.time()
    print(f"CPU Read Runtime: {end_read - start_read}")

    X = np.stack(data["feature_array"].values)
    y = data["label"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    end_etl = time.time()
    etl_runtime = end_etl - start_etl
    print(f"CPU ETL Runtime: {etl_runtime}")

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
            "tree_method": "hist",
            "device": "cpu",
        }

        model = xgb.XGBRegressor(**params)

        start_fit = time.time()
        # fit model
        model.fit(X_train, y_train, verbose=False)
        end_fit = time.time()
        fit_runtime = end_fit - start_fit
        print(f"CPU Fit Runtime: {fit_runtime}")

        # Perform predictions on CPU. Predict returns numpy array.
        start_predict = time.time()
        predictions = model.predict(X_val)
        end_predict = time.time()
        predict_runtime = end_predict - start_predict
        print(f"CPU Predict Runtime: {predict_runtime}")

        rmse = mean_squared_error(y_val, predictions, squared=False)

        return rmse

    # Load CPU study to track results
    study = optuna.load_study(
        study_name=f"optuna-spark-xgboost-cpu", storage=f"mysql://optuna_user:optuna_password@{driver_ip}/optuna"
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


total_trials = 100 # Total trials needed to run
total_tasks = 8 # Spark tasks to launch. Make sure total_tasks <= parallelism of Spark

print("\nRunning CPU Benchmark.\n")
start_cpu = time.time()
with joblib.parallel_backend("spark", n_jobs=8):
    results_cpu = joblib.Parallel()(
        joblib.delayed(task)(i) for i in partition_trials(total_trials, total_tasks)
    )
end_cpu = time.time()
cpu_runtime = end_cpu - start_cpu

best_params_cpu = min(results_cpu, key=lambda x: x[1])[0]
best_value_cpu = min(results_cpu, key=lambda x: x[1])[1]

print(f"CPU Total Runtime: {cpu_runtime:.2f} seconds")
print(f"Best parameters (CPU): {best_params_cpu}")
print(f"Best value (CPU): {best_value_cpu}")

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
import argparse
import os
import joblib
import optuna
import xgboost as xgb
from pyspark.sql import SparkSession
from pyspark import TaskContext

def task(num_trials: int, driver_ip: str, filepath: str):
    """
    This demo is to distribute xgboost training on spark cluster. 
    We implement best practices to precompute data and maximize computations on the GPU.
    Reference: https://forecastegy.com/posts/xgboost-hyperparameter-tuning-with-optuna/
    Dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
    """

    import cudf
    from cuml.metrics.regression import mean_squared_error
    from cuml.model_selection import train_test_split

    def get_gpu_id(task_context: TaskContext) -> int:
        """Get the gpu id from the task resources"""
        if task_context is None:
            # This is a safety check.
            raise RuntimeError("_get_gpu_id should not be invoked from driver side.")
        resources = task_context.resources()
        if "gpu" not in resources:
            raise RuntimeError(
                "Couldn't get the gpu id, Please check the GPU resource configuration"
            )
        # Return the first GPU ID.
        return int(resources["gpu"].addresses[0].strip())

    gpu_id = get_gpu_id(TaskContext.get())

    if filepath.startswith("/dbfs/"):
        # Check the environment variable to ensure GDS is disabled for cuDF on databricks.
        libcudf_policy = os.environ.get('LIBCUDF_CUFILE_POLICY')
        if libcudf_policy != 'OFF':
            raise RuntimeError("Set LIBCUDF_CUFILE_POLICY=OFF to read from DBFS with cuDF.")
    
    data = cudf.read_csv(filepath, delimiter=";")
    # Extract features; label is last column. 
    X = data.iloc[:, :-1].values # Tests show that converting to cupy, then to QDM, is faster than converting from cudf directly.
    y = data["quality"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # Precompute Quantile DMatrix to avoid repeated quantization every trial.
    Xy_train_qdm = xgb.QuantileDMatrix(X_train, y_train)

    # The objective function will be executed on the node for a total of "num_trials" times.
    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "verbosity": 0,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "max_bins": trial.suggest_int("max_bins", 32, 256),
            "tree_method": "gpu_hist",
            "device": f"cuda:{gpu_id}",
        }

        if "max_bins" in params:
            # If tuning the max_bins param, we must recompute the QDM every trial.
            # Use the Scikit-learn API to train the model, which requires num_estimators.
            if "num_estimators" not in params:
                params["num_estimators"] = 1000

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            booster = model.get_booster()
        else:
            # Train the model with xgb.train() API using the precomputed QDM.
            booster = xgb.train(params, dtrain=Xy_train_qdm, num_boost_round=1000)
            
        # Perform in-place predictions on GPU using the booster.
        predictions = booster.inplace_predict(X_val)
        rmse = mean_squared_error(y_val, predictions, squared=False).get()
        
        return rmse

    study = optuna.load_study(
        study_name="optuna-spark-xgboost", storage=f"mysql://optuna_user:optuna_password@{driver_ip}/optuna"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark-Optuna-XGBoost")
    parser.add_argument("--filepath", type=str, required=True, help="Absolute path to the dataset CSV file. If running on databricks, the path should begin with /dbfs/.")
    parser.add_argument("--trials", type=int, default=100, help="Total number of trials to run (default 100).")
    parser.add_argument("--tasks", type=int, default=2, help="Total number of Spark tasks to launch (default 2). This should be <= parallelism of Spark.")
    parser.add_argument("--jobs", type=int, default=8, help="Number of threads to launch Spark applications at the same time (default 8).")
    args = parser.parse_args()

    spark = SparkSession.builder.getOrCreate()

    # Get the driver IP to connect to the MySQL server.
    driver_ip = spark.conf.get("spark.driver.host")

    # Map task onto RDD partitions and collect results.
    rdd = spark.sparkContext.parallelize(partition_trials(args.trials, args.tasks), numSlices=args.tasks)
    results = rdd.map(lambda num_trials: task(num_trials, driver_ip, args.filepath)).collect()

    best_params = min(results, key=lambda x: x[1])[0]
    best_value = min(results, key=lambda x: x[1])[1]

    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")
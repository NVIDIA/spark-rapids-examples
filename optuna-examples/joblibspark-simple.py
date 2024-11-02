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
import argparse
from joblibspark import register_spark
from pyspark.sql import SparkSession


def task(num_trials: int, driver_ip: str):

    def objective(trial):
        # time.sleep(1)
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        return (x - 2) ** 2 + (y - 3) ** 2

    study = optuna.load_study(
        study_name="optuna-spark", storage=f"mysql://optuna_user:optuna_password@{driver_ip}/optuna"
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
    parser.add_argument("--trials", type=int, default=100, help="Total number of trials to run (default 100).")
    parser.add_argument("--tasks", type=int, default=2, help="Total number of Spark tasks to launch (default 2). This should be <= parallelism of Spark.")
    parser.add_argument("--jobs", type=int, default=8, help="Number of threads to launch Spark applications at the same time (default 8).")
    parser.add_argument("--localhost", action='store_true', help="Include if the MySQL server is running on localhost (e.g., in a local Standalone cluster).")
    args = parser.parse_args()
    
    spark = SparkSession.builder.getOrCreate()
    register_spark()

    if args.localhost:
        driver_ip = "localhost"
    else:
        # Get the driver IP to connect to the MySQL server.
        driver_ip = spark.conf.get("spark.driver.host")

    with joblib.parallel_backend("spark", n_jobs=args.jobs):
        results = joblib.Parallel()(
            joblib.delayed(task)(i, driver_ip) for i in partition_trials(args.trials, args.tasks)
        )

    best_params = min(results, key=lambda x: x[1])[0]
    best_value = min(results, key=lambda x: x[1])[1]

    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")

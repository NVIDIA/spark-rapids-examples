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
from typing import Iterable, List, Dict, Optional, Union, Sequence
import json
import math
import argparse
import pandas as pd
import optuna
import xgboost as xgb
from pyspark.sql import SparkSession, DataFrame
from pyspark import TaskContext
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, BooleanType
import pyspark.sql.functions as F


class OptunaParams:
    def __init__(self):
        self.hyperparameters = {}

    def add_categorical_param(self, name: str, choices: Sequence[Union[None, bool, int, float, str]]):
        """
        Adds a categorical hyperparameter to be tuned via Optuna's trial.suggest_categorical().
        """
        self.hyperparameters[name] = {
            "type": "categorical",
            "choices": choices
        }
    
    def add_int_param(self, name: str, low: int, high: int, step: int = 1, log: bool = False):
        """
        Adds an integer hyperparameter to be tuned via Optuna's trial.suggest_int().
        """
        self.hyperparameters[name] = {
            "type": "int",
            "low": low,
            "high": high,
            "step": step,
            "log": log
        }
    
    def add_float_param(self, name: str, low: float, high: float, step: Optional[float] = None, log: bool = False):
        """
        Adds a float hyperparameter to be tuned via Optuna's trial.suggest_float().
        """
        self.hyperparameters[name] = {
            "type": "float",
            "low": low,
            "high": high,
            "step": step,
            "log": log
        }

    def to_dict(self, trial) -> Dict[str, Union[int, float, str, bool]]:
        """
        Converts the hyperparameter space into a dictionary of suggested values in Optuna format,
        to be called within the objective function.
        """
        suggested_params = {}
        for name, config in self.hyperparameters.items():
            if config["type"] == "categorical":
                suggested_params[name] = trial.suggest_categorical(name, config["choices"])
            elif config["type"] == "int":
                suggested_params[name] = trial.suggest_int(
                    name, config["low"], config["high"], step=config["step"], log=config["log"]
                )
            elif config["type"] == "float":
                suggested_params[name] = trial.suggest_float(
                    name, config["low"], config["high"], step=config.get("step", None), log=config["log"]
                )
        return suggested_params

    def to_schema(self) -> StructType:
        """
        Converts the hyperparameter space into a Spark StructType output schema.
        """
        fields = []
        for name, config in self.hyperparameters.items():
            if config["type"] == "float":
                fields.append(StructField(name, DoubleType(), False))
            elif config["type"] == "int":
                fields.append(StructField(name, IntegerType(), False))
            elif config["type"] == "categorical":
                if isinstance(config["choices"][0], str):
                    fields.append(StructField(name, StringType(), False))
                elif isinstance(config["choices"][0], bool):
                    fields.append(StructField(name, BooleanType(), False))
                elif isinstance(config["choices"][0], (int, float)):
                    fields.append(StructField(name, DoubleType(), False))
                else:
                    raise ValueError(f"Unsupported categorical type for field {name}")
        fields.append(StructField("best_value", DoubleType(), False)) # Study will also return the best achieved loss.
        return StructType(fields)

    def __repr__(self):
        return f"HyperparameterSpace({self.hyperparameters})"


def task_udf(pdf_iter: Iterable[pd.DataFrame],
             hyperparams: OptunaParams,
             driver_ip: str,
             trials_per_partition: List[int]) -> Iterable[pd.DataFrame]:
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
            raise RuntimeError("_get_gpu_id should not be invoked from driver side.")
        resources = task_context.resources()
        if "gpu" not in resources:
            raise RuntimeError(
                "Couldn't get the gpu id, Please check the GPU resource configuration"
            )
        return int(resources["gpu"].addresses[0].strip()) # Return the first GPU ID for multi-GPU setups.

    tc = TaskContext.get()
    gpu_id = get_gpu_id(tc)

    # Concatenate the data batches into a cuDF DataFrame.
    df_list = []
    for pdf in pdf_iter:
        df_list.append(cudf.DataFrame.from_pandas(pdf))
    data = cudf.concat(df_list)

    # Get num trials for this partition, stored in "pid" column.
    num_trials = trials_per_partition[tc.partitionId()]
    print(f"Running {num_trials} trials on partition {tc.partitionId()}.")

    # Extract features; label is last column. 
    X = data.iloc[:, :-1].values
    y = data["quality"].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    Xy_train_qdm = xgb.QuantileDMatrix(X_train, y_train) # Precompute Quantile DMatrix to avoid repeated quantization every trial.

    def objective(trial):
        params = {
            "objective": "reg:squarederror",
            "verbosity": 0,
            "tree_method": "gpu_hist",
            "device": f"cuda:{gpu_id}",
        }
        params.update(hyperparams.to_dict(trial))

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

    result_dict = {f"{key}": [value] for key, value in study.best_params.items()}
    result_dict['best_value'] = [study.best_value]
    
    yield pd.DataFrame(result_dict)


def partition_trials(total_trials: int, total_tasks: int) -> List[int]:
    """
    Partitions the total trials into a list of trials to execute in each task.
    """
    base_size = total_trials // total_tasks
    extra = total_trials % total_tasks
    partitions = [base_size] * total_tasks
    for i in range(extra):
        partitions[i] += 1

    return partitions


def coalesce_tree_union(df: DataFrame, num_duplicates: int):
    """
    Coalesce the DataFrame to a single partition and recursively self-union to create duplicates.
    """
    input_df = df.coalesce(1).cache()
    current_df = input_df
    
    if num_duplicates <= 1:
        return current_df

    recursions = int(math.log(num_duplicates, 2))
    remainder = num_duplicates - 2 ** recursions

    for _ in range(recursions):
        current_df = current_df.union(current_df)

    for _ in range(remainder):
        current_df = current_df.union(input_df)
    
    return current_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark-Optuna-XGBoost")
    parser.add_argument("--filepath", type=str, required=True, help="Absolute path to the dataset CSV file. If running on databricks, the path should begin with /dbfs/.")
    parser.add_argument("--trials", type=int, default=100, help="Total number of trials to run (default 100).")
    parser.add_argument("--tasks", type=int, default=2, help="Number of Spark tasks to launch (default 2).")
    parser.add_argument("--localhost", action='store_true', help="Include if the MySQL server is running on localhost (e.g., in a local Standalone cluster).")

    args = parser.parse_args()
    spark = SparkSession.builder.getOrCreate()

    if args.localhost:
        driver_ip = "localhost"
    else:
        driver_ip = spark.conf.get("spark.driver.host")

    # Wine Quality Features:
    in_schema = StructType([
        StructField("fixed acidity", DoubleType(), True),
        StructField("volatile acidity", DoubleType(), True),
        StructField("citric acid", DoubleType(), True),
        StructField("residual sugar", DoubleType(), True),
        StructField("chlorides", DoubleType(), True),
        StructField("free sulfur dioxide", DoubleType(), True),
        StructField("total sulfur dioxide", DoubleType(), True),
        StructField("density", DoubleType(), True),
        StructField("pH", DoubleType(), True),
        StructField("sulphates", DoubleType(), True),
        StructField("alcohol", DoubleType(), True),
        StructField("quality", IntegerType(), True)
    ])

    num_partitions = args.tasks

    data_df = spark.read.csv(args.filepath, header=True, schema=in_schema, sep=";")
    data_df = coalesce_tree_union(data_df, num_duplicates=num_partitions)    

    # Define the params for Optuna to tune.
    hyperparams = OptunaParams()
    hyperparams.add_float_param("learning_rate", low=1e-3, high=0.1, log=True)
    hyperparams.add_int_param("max_depth", low=1, high=10)
    hyperparams.add_float_param("subsample", low=0.05, high=1.0)
    hyperparams.add_float_param("colsample_bytree", low=0.05, high=1.0)
    hyperparams.add_int_param("min_child_weight", low=1, high=20)
    hyperparams.add_int_param("max_bins", low=32, high=256)
    out_schema = hyperparams.to_schema()

    trials_per_partition = partition_trials(args.trials, args.tasks)

    result_df = data_df.mapInPandas(lambda pdf_iter: 
                                        task_udf(pdf_iter, 
                                                hyperparams=hyperparams,
                                                driver_ip=driver_ip,
                                                trials_per_partition=trials_per_partition),
                                                schema=out_schema).toPandas()
                
    results = result_df.iloc[0].to_dict()
    print("Best Results:\n", json.dumps(results, indent=4))

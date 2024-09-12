from typing import List

import os
import joblib
import optuna
import cudf
import xgboost as xgb
from joblibspark import register_spark
from pyspark import TaskContext
from cuml.metrics.regression import mean_squared_error
from cuml.model_selection import train_test_split

register_spark()

# Get the databricks driver IP to connect to the MySQL server.
driver_ip = spark.conf.get("spark.driver.host", None)

if driver_ip is None:
    raise ValueError("Driver IP could not be retrieved from Spark configuration.")

# Check the environment variable to ensure GDS is disabled for cuDF on databricks
assert(os.environ.get('LIBCUDF_CUFILE_POLICY') == 'OFF')

def task(num_trials: int = 100):
    """
    This demo is to distribute xgboost training on spark cluster by referring to
    https://forecastegy.com/posts/xgboost-hyperparameter-tuning-with-optuna/
    This code contains best practices to precompute data and read to GPU, and perform training + prediction on GPU.
    This example uses Joblib-Spark to distribute the trials across the cluster.
    """

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
        # return the first gpu id.
        return int(resources["gpu"].addresses[0].strip())

    gpu_id = get_gpu_id(TaskContext.get())
    
    url = "/dbfs/FileStore/datasets/winequality-red.csv"
    data = cudf.read_csv(url, delimiter=";")
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

# The total trials need to run
total_trials = 100
# How many spark tasks will be launched, Please make sure total_tasks should be <= parallelism of spark
total_tasks = 2

# n_jobs=8 means Spark Backend will launch at most 8 threads to launch spark applications at the same time.
with joblib.parallel_backend("spark", n_jobs=8):
    results = joblib.Parallel()(
        joblib.delayed(task)(i) for i in partition_trials(total_trials, total_tasks)
    )

best_params = min(results, key=lambda x: x[1])[0]
best_value = min(results, key=lambda x: x[1])[1]

print(f"Best parameters: {best_params}")
print(f"Best value: {best_value}")

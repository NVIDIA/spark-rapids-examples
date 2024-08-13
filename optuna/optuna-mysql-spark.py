import joblib
import optuna
from joblibspark import register_spark

register_spark()


def task(seed):
    def objective(trial):
        # time.sleep(1)
        x = trial.suggest_float('x', -10, 10)
        y = trial.suggest_float('y', -10, 10)
        return (x - 2) ** 2 + (y - 3) ** 2

    study = optuna.load_study(
        study_name="optuna-spark", storage="mysql://optuna_user:optuna_password@10.19.129.248/optuna"
    )
    study.optimize(objective, n_trials=100)
    return study.best_params, study.best_value


# Set n_jobs to the number of workers which is used to run optuna task.
# TODO, Support resource profile to get the GPU.
n_jobs = 2

with joblib.parallel_backend("spark", n_jobs=n_jobs):
    results = joblib.Parallel()(
        joblib.delayed(task)(i) for i in range(n_jobs)
    )

best_params = min(results, key=lambda x: x[1])[0]
best_value = min(results, key=lambda x: x[1])[1]

print(f"Best parameters: {best_params}")
print(f"Best value: {best_value}")

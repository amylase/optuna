import time
from typing import Callable

import optuna
from optuna import Trial
import optuna.storages
import subprocess


def ensure_mysql_instance():
    subprocess.call("docker run -d --rm -p 3306:3306 \
        -e MYSQL_USER=optuna -e MYSQL_DATABASE=optuna -e MYSQL_PASSWORD=password \
        -e MYSQL_ALLOW_EMPTY_PASSWORD=yes --name optuna-mysql mysql:8.0", shell=True)


def ensure_optuna_loaded():
    run_optimize(storage=optuna.storages.InMemoryStorage(), objective=objective, n_trials=50)


def initialize():
    ensure_mysql_instance()
    ensure_optuna_loaded()


def objective(trial: Trial) -> float:
    x = trial.suggest_float("x", -100, 100)
    y = trial.suggest_int("y", -100, 100)
    return x**2 + y**2


def run_optimize(
    storage: optuna.storages.BaseStorage,
    objective: Callable[[Trial], float],
    n_trials: int
):
    study = optuna.create_study(storage=storage)
    study.optimize(objective, n_trials=n_trials)


def run_ask_tell(
    storage: optuna.storages.BaseStorage,
    objective: Callable[[Trial], float],
    n_trials: int
):
    study = optuna.create_study(storage=storage)
    for _ in range(n_trials):
        trial = study.ask()
        value = objective(trial)
        study.tell(trial, value)


def main():
    from pathlib import Path
    import json

    initialize()

    runner = run_optimize
    runner = run_ask_tell

    storage_url = "mysql+pymysql://optuna:password@127.0.0.1:3306/optuna"
    storage = optuna.storages.RDBStorage(storage_url)
    storage = optuna.storages._CachedStorage(storage)

    n_trials = 1000

    elapsed = -time.time()
    runner(storage, objective, n_trials)
    elapsed += time.time()

    print(elapsed)
    results = {
        "timestamp": time.time(),
        "elapsed": elapsed,
        "storage": storage.__class__.__name__,
        "runner": runner.__name__,
        "n_trials": n_trials,
        "label": "v3.1.0"
    }

    results_path = Path(__file__).parent / "results.txt"
    with results_path.open("a") as f:
        print(json.dumps(results), file=f)


if __name__ == "__main__":
    main()

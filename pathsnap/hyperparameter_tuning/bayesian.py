"""Bayesian search spaces and Gaussian-process expected improvement."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from pathsnap.classifier_selection.evaluation import summarize_cv


@dataclass(frozen=True)
class Parameter:
    name: str
    low: float
    high: float
    scale: str = "linear"

    def decode(self, value):
        if self.scale == "log":
            return float(np.exp(np.log(self.low) + value * np.log(self.high / self.low)))
        decoded = self.low + value * (self.high - self.low)
        return round(decoded) if self.scale == "integer" else float(decoded)


SEARCH_SPACES = {
    "logistic_regression": (
        Parameter("C", 0.0001, 50, "log"),
    ),
    "gaussian_naive_bayes": (
        Parameter("var_smoothing", 1e-11, 0.5, "log"),
    ),
    "decision_tree": (
        Parameter("max_depth", 5, 24, "integer"),
        Parameter("min_samples_leaf", 8, 140, "integer"),
    ),
    "extra_trees": (
        Parameter("n_estimators", 28, 80, "integer"),
        Parameter("max_depth", 10, 24, "integer"),
        Parameter("min_samples_leaf", 4, 65, "integer"),
        Parameter("max_features", 0.4, 1.0),
    ),
    "hist_gradient_boosting": (
        Parameter("max_iter", 65, 150, "integer"),
        Parameter("max_leaf_nodes", 15, 63, "integer"),
        Parameter("learning_rate", 0.04, 0.23, "log"),
        Parameter("min_samples_leaf", 16, 120, "integer"),
        Parameter("l2_regularization", 0.02, 15, "log"),
    ),
}


def bayesian_search(model, search_space, evaluate, trials, seed):
    """Use a Matérn Gaussian-process surrogate and expected improvement."""
    dimensions = len(search_space)
    pool = qmc.Sobol(d=dimensions, scramble=True, seed=seed).random_base2(10)
    designs = {}
    baseline = {parameter.name: model.parameters[parameter.name] for parameter in search_space}
    for point in pool:
        params = {param.name: param.decode(unit) for param, unit in zip(search_space, point)}
        if params != baseline:
            designs.setdefault(tuple(params.values()), (point, params))
    pool = list(designs.values())
    if len(pool) < trials:
        raise ValueError("The search space has fewer unique configurations than requested trials")

    points, values, results = [], [], []
    for trial in range(1, trials + 1):
        start = perf_counter()
        acquisition = None
        if trial <= 4:
            index = 0
        else:
            surrogate = GaussianProcessRegressor(
                kernel=Matern(length_scale=0.35, nu=2.5),
                optimizer=None,
                normalize_y=True,
                alpha=1e-5,
            ).fit(points, values)
            mean, sd = surrogate.predict([point for point, _ in pool], return_std=True)
            improvement = mean - max(values) - 0.0001
            z = improvement / np.maximum(sd, 1e-15)
            expected = improvement * ndtr(z) + sd * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
            index = int(np.argmax(expected))
            acquisition = float(expected[index])
        point, parameters = pool.pop(index)
        optimizer_seconds = perf_counter() - start
        fold_rows = evaluate(model, parameters)
        summary = summarize_cv(fold_rows)
        points.append(point)
        values.append(summary["cv_macro_f1_mean"])
        result = {
            "trial": trial, "suggestion": "Sobol" if acquisition is None else "expected_improvement",
            "parameters": parameters, "expected_improvement": acquisition,
            "optimizer_seconds": optimizer_seconds, **summary,
            "folds": fold_rows,
        }
        print(f"Bayesian trial {trial:02d}/{trials}: Macro F1 {values[-1]:.5f} {parameters}", flush=True)
        results.append(result)
    return results

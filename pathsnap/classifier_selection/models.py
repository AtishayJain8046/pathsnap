"""Define and fit the five candidate classifiers."""

from dataclasses import dataclass
from time import perf_counter

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class Model:
    name: str
    family: str
    estimator: type
    parameters: dict
    scale_features: bool = False

    def create(self, seed, overrides=None):
        parameters = self.parameters | (overrides or {})
        if self.estimator is not GaussianNB:
            parameters["random_state"] = seed
        classifier = self.estimator(**parameters)
        steps = [SimpleImputer(strategy="median", keep_empty_features=True)]
        if self.scale_features:
            steps.append(StandardScaler())
        return make_pipeline(*steps, classifier)


MODELS = (
    Model("logistic_regression", "Linear", LogisticRegression,
          {"C": 1.0, "max_iter": 250, "solver": "lbfgs"}, True),
    Model("gaussian_naive_bayes", "Probabilistic", GaussianNB,
          {"var_smoothing": 1e-9}, True),
    Model("decision_tree", "Single tree", DecisionTreeClassifier,
          {"max_depth": 12, "min_samples_leaf": 30, "criterion": "gini"}),
    Model("extra_trees", "Randomized tree ensemble", ExtraTreesClassifier,
          {"n_estimators": 48, "max_depth": 16, "min_samples_leaf": 12, "max_features": 0.8, "n_jobs": 4}),
    Model("hist_gradient_boosting", "Boosted trees", HistGradientBoostingClassifier,
          {"max_iter": 100, "max_leaf_nodes": 31, "learning_rate": 0.1, "min_samples_leaf": 30,
           "l2_regularization": 1.0, "early_stopping": False}),
)


def fit_model(model, parameters, seed, features, targets):
    estimator = model.create(seed, parameters)
    start = perf_counter()
    estimator.fit(features, targets)
    return estimator, perf_counter() - start

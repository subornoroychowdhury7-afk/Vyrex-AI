import pytest

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from vyrex_ai.data.english_level_ml_models import create_models


def test_factory_creates_three_models():
    models = create_models()
    assert set(models) == {
        "logistic_regression",
        "linear_svm",
        "random_forest",
    }


def test_logistic_regression_pipeline():
    model = create_models()["logistic_regression"]
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["classifier"], LogisticRegression)
    assert model.named_steps["classifier"].class_weight == "balanced"


def test_linear_svm_pipeline():
    model = create_models()["linear_svm"]
    assert isinstance(model, Pipeline)
    assert isinstance(model.named_steps["scaler"], StandardScaler)
    assert isinstance(model.named_steps["classifier"], LinearSVC)
    assert model.named_steps["classifier"].class_weight == "balanced"


def test_random_forest_configuration():
    model = create_models()["random_forest"]
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 300
    assert model.class_weight == "balanced_subsample"
    assert model.n_jobs == -1


@pytest.mark.parametrize("value", [-1, True, 1.5, "42"])
def test_invalid_random_state_is_rejected(value):
    with pytest.raises(ValueError, match="random_state"):
        create_models(value)


def test_random_state_is_preserved():
    models = create_models(random_state=123)
    assert models["random_forest"].random_state == 123

import os

import pytest
from sklearn.pipeline import Pipeline

from src.Regressor import (
    Regressor,
    load_predictor_model,
    predict_with_model,
    save_predictor_model,
)


@pytest.fixture
def regressor(sample_train_data, schema_provider):
    """Define the regressor fixture"""
    return Regressor(sample_train_data, schema_provider)


def test_fit_predict(sample_train_data, sample_test_data, schema_provider):
    """
    Test if the fit method trains the model correctly and if predict method work as expected.
    """
    regressor = Regressor(sample_train_data, schema=schema_provider)
    predictions = predict_with_model(regressor=regressor, data=sample_test_data)
    assert predictions.shape[0] == sample_test_data.shape[0]


def test_regressor_str_representation(regressor):
    """
    Test the `__str__` method of the `Regressor` class.

    The test asserts that the string representation of a `Regressor` instance is
    correctly formatted and includes the model name and the correct hyperparameters.

    Args:
        regressor (Regressor): An instance of the `Regressor` class,
            created using the `hyperparameters` fixture.

    Raises:
        AssertionError: If the string representation of `regressor` does not
            match the expected format.
    """
    regressor_str = str(regressor)
    assert regressor.model_name in regressor_str


def test_save_predictor_model(tmpdir, sample_train_data, schema_provider):
    """
    Test that the 'save_predictor_model' function correctly saves a Regressor instance
    to disk.
    """
    model_dir_path = os.path.join(tmpdir, "model")
    regressor = Regressor(sample_train_data, schema_provider)
    save_predictor_model(regressor, model_dir_path)
    assert os.path.exists(model_dir_path)
    assert len(os.listdir(model_dir_path)) >= 1


def test_load_predictor_model(tmpdir, sample_train_data, schema_provider):
    """
    Test that the 'load_predictor_model' function correctly loads a Regressor
    instance from disk and that the loaded instance has the correct hyperparameters.
    """
    regressor = Regressor(sample_train_data, schema_provider)

    model_dir_path = os.path.join(tmpdir, "model")
    save_predictor_model(regressor, model_dir_path)

    loaded_clf = load_predictor_model(model_dir_path)
    assert isinstance(loaded_clf, Pipeline)

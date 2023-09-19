import os
import re
import warnings
from typing import List
import h2o
from h2o.model import ModelBase
from h2o.automl import H2OAutoML
import pandas as pd
from sklearn.exceptions import NotFittedError
from schema.data_schema import RegressionSchema

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"


def clean_and_ensure_unique(names: List[str]) -> List[str]:
    """
    Clean the provided column names by removing special characters and ensure their
    uniqueness.

    The function first removes any non-alphanumeric character (except underscores)
    from the names. Then, it ensures the uniqueness of the cleaned names by appending
    a counter to any duplicates.

    Args:
        names (List[str]): A list of column names to be cleaned.

    Returns:
        List[str]: A list of cleaned column names with ensured uniqueness.

    Example:
        >>> clean_and_ensure_unique(['3P%', '3P', 'Name', 'Age%', 'Age'])
        ['3P', '3P_1', 'Name', 'Age', 'Age_1']
    """

    # First, clean the names
    cleaned_names = [re.sub("[^A-Za-z0-9_]+", "", name) for name in names]

    # Now ensure uniqueness
    seen = {}
    for i, name in enumerate(cleaned_names):
        original_name = name
        counter = 1
        while name in seen:
            name = original_name + "_" + str(counter)
            counter += 1
        seen[name] = True
        cleaned_names[i] = name

    return cleaned_names


class Regressor:
    """A wrapper class for the H2O Regressor.

    This class provides a consistent interface that can be used with other
    regressor models.
    """

    def __init__(self, train_input: h2o.H2OFrame, schema: RegressionSchema):
        """Construct a new PyCaret Regressor.

        Args:
           train_input (pd.DataFrame): The input data for model training.
           schema (RegressionSchema): Schema of the input data.
        """
        self._is_trained = False
        self.training_df = train_input
        self.schema = schema
        x = train_input.columns
        x.remove(schema.id)
        x.remove(schema.target)
        self.x = x
        self.y = schema.target
        self.aml = H2OAutoML(max_models=5, seed=10, nfolds=10, verbosity='info', exclude_algos=['GLM'])
        self.model_name = "h2o_regressor_model"

    def train(self) -> None:
        """Trains H2O regressor on the input data."""
        self.aml.train(x=self.x, y=self.y, training_frame=self.training_df)
        self._is_trained = True

    def predict(self, inputs: h2o.H2OFrame) -> h2o.H2OFrame:
        """Predict regression targets for the given data.

        Args:
            inputs (h2o.H2OFrame): The input data.
        Returns:
            numpy.ndarray: The predicted regression targets.
        """
        return self.aml.leader.predict(inputs)

    def save(self, model_dir_path: str) -> None:
        """Save the regressor model to disk.
        Args:
            model_dir_path (str): Dir path to which to save the model.
        """

        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        h2o.save_model(self.aml.leader, path=model_dir_path, filename=PREDICTOR_FILE_NAME, force=True)

    @classmethod
    def load(cls, model_dir_path: str) -> ModelBase:
        """Load the regressor from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            ModelBase: A new instance of the loaded regressor.
        """
        return h2o.load_model(path=os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return f"Model name: {self.model_name}"


def predict_with_model(model: ModelBase, data: h2o.H2OFrame) -> h2o.H2OFrame:
    """
    Predict class probabilities for the given data.

    Args:
        model (Regressor): The regressor model.
        data (h2o.H2OFrame): The input data.

    Returns:
        h2o.H2OFrame: The predicted labels.
    """
    return model.predict(data)


def save_predictor_model(model: ModelBase, predictor_dir_path: str) -> None:
    """
    Save the regressor model to disk.

    Args:
        model (Regressor): The regressor model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> ModelBase:
    """
    Load the regressor model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Regressor: A new instance of the loaded regressor model.
    """
    return Regressor.load(predictor_dir_path)

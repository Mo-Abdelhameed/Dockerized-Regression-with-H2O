import h2o
import pandas as pd

from config import paths
from logger import get_logger
from Regressor import Regressor, predict_with_model
from schema.data_schema import load_saved_schema
from utils import ResourceTracker, read_csv_in_directory, save_dataframe_as_csv

logger = get_logger(task_name="predict")


def create_predictions_dataframe(
    predictions_df: h2o.H2OFrame,
    ids: h2o.H2OFrame,
) -> pd.DataFrame:
    """
    Converts the predictions numpy array into a dataframe having the required structure.

    Args:
        predictions_df (predictions_df): Predicted probabilities from predictor model.
        ids (h20.H2OFrame): identifier column of the input data.

    Returns:
        The predictions DataFrame.
    """
    predictions_df = predictions_df.cbind(ids)
    predictions_df = predictions_df.as_data_frame()
    predictions_df.rename(columns={"predict": "prediction"}, inplace=True)
    return predictions_df


def run_batch_predictions(
    test_dir=paths.TEST_DIR,
    predictor_dir=paths.PREDICTOR_DIR_PATH,
    predictions_file_path=paths.PREDICTIONS_FILE_PATH,
    saved_schema_dir=paths.SAVED_SCHEMA_DIR_PATH,
) -> None:
    """
    Run batch predictions on test data, save the predicted probabilities to a CSV file.

    This function reads test data from the specified directory,
    loads the preprocessing pipeline and pre-trained predictor model,
    transforms the test data using the pipeline,
    makes predictions using the trained predictor model,
    adds ids into the predictions dataframe,
    and saves the predictions as a CSV file.
    """
    
    with ResourceTracker(logger, monitoring_interval=0.1):
        h2o.init()
        x_test = read_csv_in_directory(test_dir)
        data_schema = load_saved_schema(saved_schema_dir)
        x_test = h2o.H2OFrame(x_test)
        ids = x_test[data_schema.id]

        for cat_columns in data_schema.categorical_features:
            x_test[cat_columns] = x_test[cat_columns].ascharacter()
            x_test[cat_columns] = x_test[cat_columns].asfactor()

        model = Regressor.load(predictor_dir)
        logger.info("Making predictions...")
        predictions_df = predict_with_model(model, x_test)

    logger.info("Saving predictions...")
    predictions_df = create_predictions_dataframe(
        predictions_df=predictions_df,
        ids=ids,
    )
    save_dataframe_as_csv(dataframe=predictions_df, file_path=predictions_file_path)
    logger.info("Batch predictions completed successfully")


if __name__ == "__main__":
    run_batch_predictions()

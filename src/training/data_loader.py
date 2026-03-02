import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)

    logger.info(f"Dataset shape: {df.shape}")

    if df.isnull().sum().sum() > 0:
        logger.warning("Dataset contains missing values.")

    return df

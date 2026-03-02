import re
import string
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)  # remove HTML
    text = re.sub(r"\S+@\S+", " ", text)  # remove emails
    text = re.sub(r"\d+", " ", text)  # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting preprocessing...")

    df = df.copy()

    # Remove nulls
    df = df.dropna(subset=["Document", "Topic_group"])

    # Remove empty documents
    df = df[df["Document"].str.strip() != ""]

    # Clean text
    df["Document"] = df["Document"].astype(str).apply(clean_text)

    logger.info(f"Dataset shape after preprocessing: {df.shape}")

    return df

from src.config import RAW_DATA_DIR
from src.training.data_loader import load_data
from src.training.preprocessing import preprocess_dataframe
from src.training.train_baseline import train_baseline
from src.training.train_bert import train_bert


def main():
    dataset_path = RAW_DATA_DIR / "helpdesk_tickets.csv"

    # Load & preprocess once
    df = load_data(dataset_path)
    df = preprocess_dataframe(df)

    # 1️⃣ Train TF-IDF baseline
    train_baseline(df)

    # 2️⃣ Train BERT
    train_bert(df)


if __name__ == "__main__":
    main()

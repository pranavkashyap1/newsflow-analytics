import os
import pandas as pd

def setup():
    if not os.path.exists("data/processed/articles.csv"):
        print("Generating processed dataset...")
        from src.data_loader import load_raw_data
        from src.preprocessor import process_and_save
        df = load_raw_data()
        process_and_save(df)
        print("Dataset ready.")

    if not os.path.exists("models/rf_regressor.pkl"):
        print("Training ML models...")
        df = pd.read_csv("data/processed/articles.csv", quoting=1)
        from src.ml_models import train_models
        from src.nlp_analyzer import train_nlp_model
        train_models(df)
        train_nlp_model(df)
        print("Models ready.")

if __name__ == "__main__":
    setup()

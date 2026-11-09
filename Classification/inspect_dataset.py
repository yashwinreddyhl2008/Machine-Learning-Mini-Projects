from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Phishing_Legitimate_full.csv"

df = pd.read_csv(DATA_PATH)
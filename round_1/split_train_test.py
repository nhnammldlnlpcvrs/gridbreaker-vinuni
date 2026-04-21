from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_CSV = BASE_DIR / "Analytical" / "sales.csv"
TRAIN_CSV = BASE_DIR / "Analytical" / "sales_train.csv"
TEST_CSV = BASE_DIR / "Analytical" / "sales_test.csv"

TRAIN_START = pd.Timestamp("2012-07-04")
TRAIN_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2024-07-01")


def find_date_column(df: pd.DataFrame) -> str:
	for col in df.columns:
		lower = col.strip().lower()
		if lower in {"date", "day", "order_date", "created_at", "timestamp"}:
			return col
	return df.columns[0]


def main() -> None:
	df = pd.read_csv(INPUT_CSV)
	date_col = find_date_column(df)

	df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
	df = df.dropna(subset=[date_col])

	train_df = df[(df[date_col] >= TRAIN_START) & (df[date_col] <= TRAIN_END)]
	test_df = df[(df[date_col] >= TEST_START) & (df[date_col] <= TEST_END)]

	train_df.to_csv(TRAIN_CSV, index=False)
	test_df.to_csv(TEST_CSV, index=False)


if __name__ == "__main__":
	main()

import re
import pandas as pd
from pathlib import Path


def parse_market_value_to_eur(val) -> float:

    if val is None:
        return 0.0

    s = str(val).strip()

    if s in {"", "-", "Na", "nan", "None"}:
        return 0.0

    s = s.replace("€", "").replace(" ", "").lower()

    s = s.replace(",", "")

    m = re.match(r"^([0-9]*\.?[0-9]+)(bn|m|k)?$", s)
    if not m:
        return 0.0

    number = float(m.group(1))
    suffix = m.group(2)

    multiplier = 1.0
    if suffix == "k":
        multiplier = 1_000.0
    elif suffix == "m":
        multiplier = 1_000_000.0
    elif suffix == "bn":
        multiplier = 1_000_000_000.0

    return number * multiplier


def prepare_transfermarkt_dataset(
    input_csv: str = "data/scraped-football-player-market-values/Players_transfer_market_data_complete.csv",
    output_csv: str = "data/scraped-football-player-market-values/Players_transfer_market_data_complete_prepared.csv",
) -> None:

    input_path = Path(input_csv)
    output_path = Path(output_csv)

    df = pd.read_csv(input_path)

    df = df.replace("Na", pd.NA)

    drop_cols = ["Link", "Place of birth", "Player agent"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    if "Market value" not in df.columns:
        raise ValueError("Expected column 'Market value' not found in CSV.")

    df["Market_value_eur"] = df["Market value"].apply(parse_market_value_to_eur)

    df = df.drop(columns=["Market value"], errors="ignore")

    df = df.fillna(0)

    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved prepared dataset to: {output_path}")
    print("Shape:", df.shape)
    print("Market_value_eur summary:")
    print(df["Market_value_eur"].describe())


if __name__ == "__main__":
    prepare_transfermarkt_dataset()

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


def extract_age_years(val) -> float:
    if pd.isna(val):
        return 0.0
    s = str(val)
    m = re.search(r"\((\d+)\)", s)
    return float(m.group(1)) if m else 0.0


def parse_date_ddmmyyyy(val):
    if val is None or pd.isna(val):
        return pd.NaT
    s = str(val).strip()
    if s in {"", "-", "Na"}:
        return pd.NaT
    return pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")


def months_between(start_date, end_date) -> float:
    if pd.isna(start_date) or pd.isna(end_date):
        return 0.0
    days = (end_date - start_date).days
    return max(0.0, days / 30.44)


def parse_height_to_meters(val) -> float:
    """
    Robust parser for heights like '1,96 m', '1,85\xa0m'
    """
    if val is None or pd.isna(val):
        return 0.0

    s = str(val).lower()

    # remove non-breaking spaces and normal spaces
    s = s.replace("\xa0", "").replace(" ", "")

    # remove unit
    s = s.replace("m", "")

    # normalize decimal separator
    s = s.replace(",", ".")

    try:
        return float(s)
    except ValueError:
        return 0.0


def prepare_transfermarkt_dataset(
    input_csv: str = "data/scraped-football-player-market-values/Players_transfer_market_data_complete.csv",
    output_csv: str = "data/scraped-football-player-market-values/Players_transfer_market_data_complete_prepared.csv",
) -> None:
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    df = pd.read_csv(input_path)

    # Standardize the scraper's "Na"
    df = df.replace("Na", pd.NA)

    # Drop columns you said to remove
    drop_cols = ["Link", "Place of birth", "Player agent"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # ---- Feature engineering: Height ----
    if "Height" in df.columns:
        df["Height_m"] = df["Height"].apply(parse_height_to_meters)
        df = df.drop(columns=["Height"], errors="ignore")

    # ---- Target: Market value -> EUR numeric ----
    if "Market value" not in df.columns:
        raise ValueError("Expected column 'Market value' not found in CSV.")
    df["Market_value_eur"] = df["Market value"].apply(parse_market_value_to_eur)
    df = df.drop(columns=["Market value"], errors="ignore")

    # ---- Feature engineering: Age from 'Date of birth/Age' ----
    if "Date of birth/Age" in df.columns:
        df["Age_years"] = df["Date of birth/Age"].apply(extract_age_years)
        # Optional: keep birthdate too
        df["Birthdate"] = df["Date of birth/Age"].astype(str).str.split(" ").str[0]
        df["Birthdate"] = pd.to_datetime(df["Birthdate"], format="%d/%m/%Y", errors="coerce")

    # ---- Feature engineering: Contract expires -> months remaining ----
    if "Contract expires" in df.columns:
        df["Contract_expires_date"] = df["Contract expires"].apply(parse_date_ddmmyyyy)
        today = pd.Timestamp.today().normalize()
        df["Contract_months_remaining"] = df["Contract_expires_date"].apply(lambda d: months_between(today, d))

    # Replace remaining missing with 0 (but keep datetimes as-is)
    # We'll fill non-datetime columns with 0 to be safe for modeling
    for col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved prepared dataset to: {output_path}")
    print("Shape:", df.shape)
    if "Market_value_eur" in df.columns:
        print("\nMarket_value_eur summary:")
        print(df["Market_value_eur"].describe())
    if "Age_years" in df.columns:
        print("\nAge_years summary:")
        print(df["Age_years"].describe())
    if "Contract_months_remaining" in df.columns:
        print("\nContract_months_remaining summary:")
        print(df["Contract_months_remaining"].describe())



if __name__ == "__main__":
    prepare_transfermarkt_dataset()

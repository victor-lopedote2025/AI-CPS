import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


TRAIN_DIR = Path("data/scraped-football-player-market-values/train")
TEST_DIR = Path("data/scraped-football-player-market-values/test")
ACTIVATION_DIR = Path("data/scraped-football-player-market-values/activation")

RANDOM_STATE = 42
TEST_SIZE = 0.2
HIGH_VALUE_QUANTILE = 0.75

SCALER = StandardScaler()

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

def parse_minutes_to_minutes(val) -> float:
    """
    Robust parser for heights like '1,96 m', '1,85\xa0m'
    """
    if val is None or pd.isna(val):
        return 0.0

    s = str(val).lower()

    # remove non-breaking spaces and normal spaces
    s = s.replace("\xa0", "").replace(" ", "")

    # remove unit
    s = s.replace("'", "")

    # normalize decimal separator
    s = s.replace(".", "")

    try:
        return float(s)
    except ValueError:
        return 0.0


def prepare_transfermarkt_dataset(
    input_csv: str = "data/scraped-football-player-market-values/Players_transfer_market_data_complete.csv",
    output_csv: str = "data/scraped-football-player-market-values/Players_transfer_market_data_complete_prepared.csv",
) -> None:
    input_path = Path("data/scraped-football-player-market-values/Players_transfer_market_data_complete.csv")
    output_path = Path("data/scraped-football-player-market-values/Players_transfer_market_data_complete_prepared.csv")

    df = pd.read_csv(input_path)

    # Standardize the scraper's "Na"
    df = df.replace("Na", pd.NA)

    # Drop unnecessary columns
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

    # ---- Feature engineering: Contract expires -> months remaining ----
    if "Contract expires" in df.columns:
        df["Contract_expires_date"] = df["Contract expires"].apply(parse_date_ddmmyyyy)
        today = pd.Timestamp.today().normalize()
        df["Contract_months_remaining"] = df["Contract_expires_date"].apply(lambda d: months_between(today, d))
        df = df.drop(columns=["Contract_expires_date"], errors="ignore")
    
    # ---- Target: Minutes value -> Minutes numeric ----
    if "Competition_Total_Minutes_Played" in df.columns:
        df["Competition_Total_Minutes_Played_m"] = df["Competition_Total_Minutes_Played"].apply(parse_minutes_to_minutes)
        df = df.drop(columns=["Competition_Total_Minutes_Played"], errors="ignore")
    
    if "Competition_International_Minutes_Played" in df.columns:
        df["Competition_International_Minutes_Played_m"] = df["Competition_International_Minutes_Played"].apply(parse_minutes_to_minutes)
        df = df.drop(columns=["Competition_International_Minutes_Played"], errors="ignore")
    
    if "Competition_Nationalcup_Minutes_Played" in df.columns:
        df["Competition_Nationalcup_Minutes_Played_m"] = df["Competition_Nationalcup_Minutes_Played"].apply(parse_minutes_to_minutes)
        df = df.drop(columns=["Competition_Nationalcup_Minutes_Played"], errors="ignore")
    
    if "Competition_National_Minutes_Played" in df.columns:
        df["Competition_National_Minutes_Played_m"] = df["Competition_National_Minutes_Played"].apply(parse_minutes_to_minutes)
        df = df.drop(columns=["Competition_National_Minutes_Played"], errors="ignore")

    # Replace remaining missing with 0 (but keep datetimes as-is)
    # We'll fill non-datetime columns with 0 to be safe for modeling
    for col in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(0)


    #Scaling of numerical features (excluding Market value)
    cols_to_be_scaled = df.select_dtypes(include=["float64", "int64"]).columns
    #cols_without_target_col = [col for col in cols_to_be_scaled if col != "Market_value_eur"]

    #Converting Object types to avoid errors in model.fit()
    categorical_cols = df.select_dtypes(include=["object"]).columns
    unnecessary_categorical_data = [col for col in categorical_cols if col not in ["Club", "Citizenship", "Foot" ,"Position"]]
    df = df.drop(columns=unnecessary_categorical_data, errors="ignore")
    df = pd.get_dummies(df, columns=["Club", "Citizenship", "Foot" ,"Position"], drop_first=True)

    df[cols_to_be_scaled] = SCALER.fit_transform(df[cols_to_be_scaled])

    #Drop unnecessary columns
    df = df.drop(columns=["Date of birth/Age"], errors="ignore")


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
    
    return df

def split_data_into_test_and_train_csv(df):

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=["Market_value_eur"])
    y = df["Market_value_eur"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    X_train.assign(Market_value_eur=y_train).to_csv(TRAIN_DIR / "training_data_set.csv", index=False)
    X_test.assign(Market_value_eur=y_test).to_csv(TEST_DIR / "test_data_set.csv", index=False)
    activation_sample = X_test.sample(1, random_state=RANDOM_STATE)
    activation_sample.to_csv(ACTIVATION_DIR / "activation_data.csv", index=False)


if __name__ == "__main__":
    split_data_into_test_and_train_csv(prepare_transfermarkt_dataset())
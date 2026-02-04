import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

TRAIN_CSV_PATH = Path("data/scraped-football-player-market-values/train/training_data_set.csv")
TEST_CSV_PATH = Path("data/scraped-football-player-market-values/test/test_data_set.csv")
MODEL_FOLDER_PATH = Path("code/Football_player_market_values_predictions/models")

def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    for col in df2.columns:
        # If boolean dtype, cast directly
        if pd.api.types.is_bool_dtype(df2[col]):
            df2[col] = df2[col].astype(int)
            continue

        # If object dtype, try to coerce:
        if df2[col].dtype == "object":
            # handle string "True"/"False" without global replace
            df2[col] = df2[col].astype(str).str.strip()
            df2[col] = df2[col].replace({"True": "1", "False": "0"})

            df2[col] = pd.to_numeric(df2[col], errors="coerce")

        # If already numeric, keep as-is

    return df2.fillna(0)

def plot_residual_plot(y_pred, residuals):
    plt.figure(figsize=(10,10))
    plt.scatter(y_pred, residuals, alpha=0.2)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Market Value")
    plt.ylabel("Residual")
    plt.title("ANN Residual Plot")
    plt.grid(True)
    plt.tight_layout()

    MODEL_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_FOLDER_PATH / "ols_residual_plot.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Prediction plot saved to: {out_path}")

def plot_residual_distribution(residuals):
    plt.figure(figsize=(10,10))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("ANN Residual Distribution")
    plt.grid(True)
    plt.tight_layout()

    MODEL_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_FOLDER_PATH / "ols_residual_distribution.pdf"
    plt.savefig(out_path)
    plt.close()

def plot_predictions_vs_true(y_true, y_pred, out_path: Path):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    plt.xlabel("True Market Value (scaled)")
    plt.ylabel("Predicted Market Value (scaled)")
    plt.title("OLS Predictions vs True Values (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    X_train = train_df.drop(columns=["Market_value_eur"])
    y_train = pd.to_numeric(train_df["Market_value_eur"], errors="coerce").fillna(0).astype(float)

    X_test = test_df.drop(columns=["Market_value_eur"])
    y_test = pd.to_numeric(test_df["Market_value_eur"], errors="coerce").fillna(0).astype(float)

    X_train = coerce_numeric(X_train)
    X_test = coerce_numeric(X_test)


    # Add intercept
    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_test_const = sm.add_constant(X_test, has_constant="add")

    X_train_const = X_train_const.astype(float)
    X_test_const = X_test_const.astype(float)


    model = sm.OLS(y_train, X_train_const).fit()
    print(model.summary())

    y_pred = model.predict(X_test_const)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    MODEL_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    # Save model as pickle
    model_path = MODEL_FOLDER_PATH / "currentOlsSolution.pickle"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved OLS model to: {model_path}")

    # Save plot
    plot_path = MODEL_FOLDER_PATH / "ols_predictions_vs_true_test.pdf"
    residuals = y_test - y_pred
    plot_residual_plot(y_test, y_pred)
    plot_residual_distribution(residuals)
    plot_predictions_vs_true(y_test.values, y_pred.values, plot_path)
    print(f"Saved OLS plot to: {plot_path}")

if __name__ == "__main__":
    main()

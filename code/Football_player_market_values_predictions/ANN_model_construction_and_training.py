import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from pathlib import Path


TRAIN_CSV_PATH = Path("data/scraped-football-player-market-values/train/training_data_set.csv")
TEST_CSV_PATH = Path("data/scraped-football-player-market-values/test/test_data_set.csv")
MODEL_FOLDER_PATH = Path("code/Football_player_market_values_predictions/models")

LEARNING_RATE = 0.0001
RANDOM_SEED = 42

EPOCHS = 1000         
BATCH_SIZE = 32


np.random.seed(RANDOM_SEED)
tensorflow.random.set_seed(RANDOM_SEED)

TRAIN_DF = pd.read_csv(TRAIN_CSV_PATH)
TEST_DF = pd.read_csv(TEST_CSV_PATH)

X_TRAIN = TRAIN_DF.drop(columns=["Market_value_eur"])
Y_TRAIN = TRAIN_DF["Market_value_eur"]
X_TEST = TEST_DF.drop(columns=["Market_value_eur"])
Y_TEST = TEST_DF["Market_value_eur"]

def build_model():
    print("Loading data...")

    print(f"Training samples: {X_TRAIN.shape}")
    print(f"Testing samples: {X_TEST.shape}")

    model = Sequential([
        Dense(256, activation="relu", input_shape=(X_TRAIN.shape[1],)),
        Dense(128, activation="relu"),
        Dense(1, activation="linear")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )


    model.summary()
    return model

def train_model(model):

    print(f"Training samples: {X_TRAIN.shape}")
    print(f"Testing samples: {X_TEST.shape}")

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("Starting training...")
    model.fit(
        X_TRAIN,
        Y_TRAIN,
        validation_data=(X_TEST, Y_TEST),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    print("Evaluating model...")    
    y_pred = model.predict(X_TEST).flatten()

    mae = mean_absolute_error(Y_TEST, y_pred)
    rmse = np.sqrt(mean_squared_error(Y_TEST, y_pred))
    r2 = r2_score(Y_TEST, y_pred)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    # Plot predictions vs true values
    plot_predictions_vs_true(Y_TEST.values, y_pred)

    # Saving the model
    MODEL_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FOLDER_PATH / "currentAiSolution.h5")
    
    print("Model saved!")


def plot_predictions_vs_true(y_true, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Reference line y = x
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")

    plt.xlabel("True Market Value (scaled)")
    plt.ylabel("Predicted Market Value (scaled)")
    plt.title("ANN Predictions vs True Values (Test Set)")
    plt.legend()
    plt.tight_layout()

    MODEL_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_FOLDER_PATH / "ann_predictions_vs_true_test.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"Prediction plot saved to: {out_path}")



if __name__ == "__main__":
    train_model(build_model())

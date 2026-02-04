import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import StandardScaler

TRAIN_CSV_PATH = Path("data/scraped-football-player-market-values/train/training_data_set.csv")
TEST_CSV_PATH = Path("data/scraped-football-player-market-values/test/test_data_set.csv")
MODEL_FOLDER_PATH = Path("code/Football_player_market_values_predictions/models")

LEARNING_RATE = 0.0007
RANDOM_SEED = 42

EPOCHS = 1000       
BATCH_SIZE = 32
X_SCALER = StandardScaler()

np.random.seed(RANDOM_SEED)
tensorflow.random.set_seed(RANDOM_SEED)

TRAIN_DF = pd.read_csv(TRAIN_CSV_PATH)
TEST_DF = pd.read_csv(TEST_CSV_PATH)

X_TRAIN = TRAIN_DF.drop(columns=["Market_value_eur"])
Y_TRAIN = TRAIN_DF["Market_value_eur"].values
X_TEST = TEST_DF.drop(columns=["Market_value_eur"])
Y_TEST = TEST_DF["Market_value_eur"].values

X_TRAIN = X_SCALER.fit_transform(X_TRAIN)
X_TEST  = X_SCALER.transform(X_TEST)

#Y_TRAIN = X_SCALER.fit_transform(Y_TRAIN.reshape(-1,1))
#Y_TEST  = X_SCALER.transform(Y_TEST.reshape(-1,1))

Y_TRAIN_USED = Y_TRAIN
Y_TEST_USED  = Y_TEST


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

def plot_loss(history):
    plt.plot(history.history['loss'], 'o', label="Training Loss")
    plt.plot(history.history['val_loss'], 'b', label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("ANN Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    MODEL_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_FOLDER_PATH / "ann_loss_curve.pdf"
    plt.savefig(out_path)
    plt.close()

def plot_mae(history):
    plt.plot(history.history['mae'], 'o', label="Training MAE")
    plt.plot(history.history['val_mae'], 'b', label="Validation MAE")
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title("ANN MAE Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    MODEL_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_FOLDER_PATH / "ann_mae_curve.pdf"
    plt.savefig(out_path)
    plt.close()

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
    out_path = MODEL_FOLDER_PATH / "ann_residual_plot.pdf"
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
    out_path = MODEL_FOLDER_PATH / "ann_residual_distribution.pdf"
    plt.savefig(out_path)
    plt.close()


def build_model(X_TEST, X_TRAIN):
    print("Loading data...")

    print(f"Training samples: {X_TRAIN.shape}")
    print(f"Testing samples: {X_TEST.shape}")

    model = Sequential([
        Dense(256, activation='relu',
              kernel_regularizer=l2(0.0005),
              input_shape=(X_TRAIN.shape[1],)),
        BatchNormalization(),
        Dropout(0.35),
    
        Dense(128, activation='relu', kernel_regularizer=l2(0.0005)),
        BatchNormalization(),
        Dropout(0.25),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    model.summary()
    return model

def train_model(model, X_TEST, X_TRAIN, Y_TRAIN, Y_TEST):

    
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=50,
        restore_best_weights=True
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )

    print(f"Training samples: {X_TRAIN.shape}")
    print(f"Testing samples: {X_TEST.shape}")

    print("Starting training...")
    history = model.fit(
        X_TRAIN,
        Y_TRAIN,
        validation_data=(X_TEST, Y_TEST),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop,lr_scheduler],
        verbose=1
    )

    plot_loss(history)
    plot_mae(history)

    print("Evaluating model...")

    y_pred = model.predict(X_TEST).flatten()
    
    mae = mean_absolute_error(Y_TEST_USED, y_pred)
    rmse = np.sqrt(mean_squared_error(Y_TEST_USED, y_pred))
    r2 = r2_score(Y_TEST_USED, y_pred)

    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")
    
    residuals = Y_TEST_USED - y_pred

    plot_predictions_vs_true(Y_TEST_USED, y_pred)
    plot_residual_plot(y_pred, residuals)
    plot_residual_distribution(residuals)

    # Saving the model
    MODEL_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FOLDER_PATH / "currentAiSolution.h5")
    
    print("Model saved!")


if __name__ == "__main__":
    train_model(build_model(X_TEST, X_TRAIN), X_TEST, X_TRAIN, Y_TRAIN, Y_TEST)

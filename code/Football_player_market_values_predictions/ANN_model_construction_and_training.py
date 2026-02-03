import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
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

    X_TRAIN = TRAIN_DF.drop(columns=["Market_value_eur"])

    X_TEST = TEST_DF.drop(columns=["Market_value_eur"])

    print(f"Training samples: {X_TRAIN.shape}")
    print(f"Testing samples: {X_TEST.shape}")

    model = Sequential([
        Dense(256, activation="sigmoid", input_shape=(X_TRAIN.shape[1],)),
        Dense(128, activation="sigmoid"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tensorflow.keras.metrics.AUC(name="auc")
        ]
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
    #Needs to be implemented
    y_pred = model.predict(X_TEST)

    #y_pred = (y_pred_prob >= 0.5).astype(int)
    # print("\nClassification Report:")
    # print(classification_report(Y_TEST, y_pred, zero_division=0.0))

    # Saving the model
    MODEL_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FOLDER_PATH / "currentAiSolution.h5")
    
    print("Model saved!")



if __name__ == "__main__":
    train_model(build_model())
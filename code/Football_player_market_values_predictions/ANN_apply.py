import pandas as pd
import tensorflow as tf
from keras.losses import MeanSquaredError
import sys

path_to_model = sys.argv[0]
path_to_activation = sys.argv[1]
X_ACTIV = pd.read_csv(path_to_activation)

new_model = tf.keras.models.load_model(path_to_model, custom_objects={ "mse": MeanSquaredError() })
y_pred = new_model.predict(X_ACTIV)

print(y_pred)
import pandas as pd
import sys
import pickle

path_to_model = sys.argv[0]
path_to_activation = sys.argv[1]
X_ACTIV = pd.read_csv(path_to_activation)

new_model = pickle.load(open(path_to_model, 'rb'))
y_pred = new_model.predict(X_ACTIV)

print(y_pred)
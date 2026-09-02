import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")
y_test = y_test["silica_concentrate"]

file = open("models/gbr_model.pkl", "rb")
model = pickle.load(file)
file.close()

predictions = model.predict(X_test_scaled)

# we save the real values and the predictions in a new dataset
prediction_df = pd.DataFrame()
prediction_df["silica_concentrate"] = y_test.values
prediction_df["prediction"] = predictions
prediction_df.to_csv("data/prediction.csv", index=False)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

scores = {"mse": mse, "r2": r2}

file = open("metrics/scores.json", "w")
json.dump(scores, file)
file.close()

print("mse :", mse)
print("r2 :", r2)

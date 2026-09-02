import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor

X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")
y_train = y_train["silica_concentrate"]

file = open("models/best_params.pkl", "rb")
best_params = pickle.load(file)
file.close()

model = GradientBoostingRegressor(
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    random_state=42,
)
model.fit(X_train_scaled, y_train)

file = open("models/gbr_model.pkl", "wb")
pickle.dump(model, file)
file.close()

print("model trained and saved")

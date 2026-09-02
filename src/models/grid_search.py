import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

y_train = y_train["silica_concentrate"]

parameters = {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}

model = GradientBoostingRegressor(random_state=42)
grid = GridSearchCV(model, parameters, cv=3, scoring="neg_mean_squared_error")
grid.fit(X_train_scaled, y_train)

print("best parameters :", grid.best_params_)

file = open("models/best_params.pkl", "wb")
pickle.dump(grid.best_params_, file)
file.close()

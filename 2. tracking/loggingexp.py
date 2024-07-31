import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.sklearn.autolog()

# mlflow.set_tracking_uri(uri='sqlite:///mlruns.db')

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)
print('Predictions:', predictions)

# add metrics
_mse = ((predictions - y_test) ** 2).mean()
mlflow.log_metric('mse', _mse)
print('MSE:', _mse)
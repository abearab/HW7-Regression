"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

# (you will probably need to import more things here)


def test_prediction():
	# Load data
	X_train, X_val, y_train, y_val = _load_data()
    # Scale the data, since values vary across feature. Note that we
    # fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	# Create a logistic regression model from sklearn
	sklearn_model = LogisticRegression(solver='saga',max_iter=1000,random_state=42)
	sklearn_model.fit(X_train, y_train)
	sklearn_pred = sklearn_model.predict(X_val)
	sklearn_acc = accuracy_score(y_val, sklearn_pred)
	
	# Create a logistic regression model from scratch
	log_model = logreg.LogisticRegressor(num_feats=X_train.shape[1], max_iter=1000, learning_rate=0.1, batch_size=1000)
	log_model.train_model(X_train, y_train, X_val, y_val)
	log_model_pred = log_model.make_prediction(X_val)
	# convert the predicted values to 0 or 1
	log_model_pred = np.where(log_model_pred > 0.5, 1, 0)
	log_model_acc = accuracy_score(y_val, log_model_pred)

	# Check that the predictions of the two models are similar
	assert np.corrcoef(sklearn_pred, log_model_pred)[0, 1] > 0.9, "predictions of the two models are not correlated"

	# Check that the accuracy of the two models is similar
	assert np.isclose(sklearn_acc, log_model_acc, atol=0.1), "accuracy of the two models are not similar"


def test_loss_function():
	# Load data
	X_train, X_val, y_train, y_val = _load_data()
	# Scale the data, since values vary across feature. Note that we
	# fit on the training data and use the same scaler for X_val.
	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)

	# Create a logistic regression model from sklearn
	sklearn_model = LogisticRegression(solver='saga',max_iter=1000,random_state=42)
	sklearn_model.fit(X_train, y_train)
	sklearn_pred = sklearn_model.predict(X_val)
	sklearn_loss = log_loss(y_val, sklearn_pred)

	# Create a logistic regression model from scratch
	log_model = logreg.LogisticRegressor(num_feats=X_train.shape[1], max_iter=1000, learning_rate=0.1, batch_size=1000)
	log_model.train_model(X_train, y_train, X_val, y_val)
	log_model_pred = log_model.make_prediction(X_val)
	# convert the predicted values to 0 or 1
	log_model_pred = np.where(log_model_pred > 0.5, 1, 0)
	log_model_loss = log_model.loss_function(y_val, log_model_pred)

	# Check that the loss of the two models is similar
	assert np.isclose(sklearn_loss, log_model_loss, atol=0.5), \
	"loss of the implemented model is not sufficiently close to the sklearn model"


def test_gradient():
	pass


def test_training():
	pass


def _load_data():
	# Load data
	X_train, X_val, y_train, y_val = utils.loadDataset(
        features=[
            'Penicillin V Potassium 500 MG',
            'Computed tomography of chest and abdomen',
            'Plain chest X-ray (procedure)',
            'Low Density Lipoprotein Cholesterol',
            'Creatinine',
            'AGE_DIAGNOSIS'
        ],
        split_percent=0.8,
        split_seed=42
	)

	return X_train, X_val, y_train, y_val

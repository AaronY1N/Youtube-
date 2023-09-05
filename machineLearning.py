from re import M
import numpy as np
from numpy.lib.function_base import gradient
import requests
import os 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pyvizml import CreateNBAData, GradientDescent
import csv

class GradientDescent:
    """
    This class defines the vanilla gradient descent algorithm for linear regression.
    Args:
        fit_intercept (bool): Whether to add intercept for this model.
    """
    def __init__(self, fit_intercept=True):
        self._fit_intercept = fit_intercept
    def find_gradient(self):
        """
        This function returns the gradient given certain model weights.
        """
        y_hat = np.dot(self._X_train, self._w)
        gradient = (2/self._m) * np.dot(self._X_train.T, y_hat - self._y_train)
        return gradient
    def mean_squared_error(self):
        """
        This function returns the mean squared error given certain model weights.
        """
        y_hat = np.dot(self._X_train, self._w)
        mse = ((y_hat - self._y_train).T.dot(y_hat - self._y_train)) / self._m
        return mse
    def fit(self, X_train, y_train, epochs=10000, learning_rate=0.001):
        """
        This function uses vanilla gradient descent to solve for weights of this model.
        Args:
            X_train (ndarray): 2d-array for feature matrix of training data.
            y_train (ndarray): 1d-array for target vector of training data.
            epochs (int): The number of iterations to update the model weights.
            learning_rate (float): The learning rate of gradient descent.
        """
        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
        self._m = self._X_train.shape[0]
        if self._fit_intercept:
            X0 = np.ones((self._m, 1), dtype=float)
            self._X_train = np.concatenate([X0, self._X_train], axis=1)
        n = self._X_train.shape[1]
        self._w = np.random.rand(n)
        n_prints = 10
        print_iter = epochs // n_prints
        w_history = dict()
        for i in range(epochs):
            current_w = self._w.copy()
            w_history[i] = current_w
            mse = self.mean_squared_error()
            gradient = self.find_gradient()
            if i % print_iter == 0:
                print("epoch: {:6} - loss: {:.6f}".format(i, mse))
            self._w -= learning_rate*gradient
        w_ravel = self._w.copy().ravel()
        self.intercept_ = w_ravel[0]
        self.coef_ = w_ravel[1:]
        self._w_history = w_history
        
class AdaGrad(GradientDescent):
    """
    This class defines the Adaptive Gradient Descent algorithm for linear regression.
    """
    def fit(self, X_train, y_train, epochs=10000, learning_rate=0.01, epsilon=1e-06):
        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
        self._m = self._X_train.shape[0]
        if self._fit_intercept:
            X0 = np.ones((self._m, 1), dtype=float)
            self._X_train = np.concatenate([X0, self._X_train], axis=1)
        n = self._X_train.shape[1]
        self._w = np.random.rand(n)
        # 初始化 ssg
        ssg = np.zeros(n, dtype=float)
        n_prints = 10
        print_iter = epochs // n_prints
        w_history = dict()
        for i in range(epochs):
            current_w = self._w.copy()
            w_history[i] = current_w
            mse = self.mean_squared_error()
            gradient = self.find_gradient()
            ssg += gradient**2
            ada_grad = gradient / (epsilon + ssg**0.5)
            if i % print_iter == 0:
                print("epoch: {:6} - loss: {:.6f}".format(i, mse))
            # 以 adaptive gradient 更新 w
            self._w -= learning_rate*ada_grad
        w_ravel = self._w.copy().ravel()
        self.intercept_ = w_ravel[0]
        self.coef_ = w_ravel[1:]
        self._w_history = w_history        


    def predict(self, X_test):
        """
        This function returns predicted values with weights of this model.
        Args:
            X_test (ndarray): 2d-array for feature matrix of test data.
        """
        self._X_test = X_test
        m = self._X_test.shape[0]
        if self._fit_intercept:
            X0 = np.ones((m, 1), dtype=float)
            self._X_test = np.concatenate([X0, self._X_test], axis=1)
        y_pred = np.dot(self._X_test, self._w)
        return y_pred

days = np.empty(shape=1)
views = np.empty(shape=1)        
with open("S:\\TaiwanSubTop100_1test.csv") as csvfile:
    data = csv.reader(csvfile)
    for rows in data:
        days = np.append(days,rows[7])
        views = np.append(views,rows[8])
days = np.delete(days,[0,1])
views = np.delete(views,[0,1])
'''星期與觀賞人數預測'''
X = days.astype(float).reshape(-1,1)
y = views.astype(float)

X_train,X_vaild,y_train,y_vaild = train_test_split(X,y,test_size=0.33,random_state=42)
h = AdaGrad()
h.fit(X_train, y_train, epochs=400000, learning_rate=100)
print(h.intercept_) # 截距項
print(h.coef_)      # 係數項





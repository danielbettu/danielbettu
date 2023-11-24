# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:41:38 2023

@author: 02901717926
"""

# regressao.py

from sklearn.linear_model import LinearRegression

def perform_linear_regression(dados):
    # Realizar a regressão linear
    x = dados.loc[:, 0].values.reshape(-1, 1)  # valores converte em numpy array, -1 significa que calcula a dimensão de linhas, mas tem uma coluna
    y = dados.loc[:, 1].values.reshape(-1, 1)  # -1 significa que calcula a dimensão de linhas, mas tem uma coluna

    linear_regressor = LinearRegression()  # criar objeto para a classe
    linear_regressor.fit(x, y)  # realizar a regressão linear
    y_pred = linear_regressor.predict(x)  # fazer as previsões

  
    
    return x, y_pred, linear_regressor

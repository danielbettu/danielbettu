# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 18:19:58 2023

@author: 02901717926
"""



def perform_gardner_correl(dados, coef_a, coef_b):
    # Realizar a regressão linear
    x = dados  # valores converte em numpy array, -1 significa que calcula a dimensão de linhas, mas tem uma coluna

    dens_formacao = coef_a * (1000000/dados) ** coef_b
  
    return dens_formacao
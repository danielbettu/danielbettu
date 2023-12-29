#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 11:55:13 2023

@author: bettu
"""

# Exercícios resolvidos do livro "Análise Estatísitica
# de dados Geológicos Multivariados" de P.M.B. Landim

import pandas as pd
import numpy as np

# pág 21 - operações com matrizes
# matriz transposta
matriz_A = [[33,48,63],[28,40,55],[12,78,93],[45,89,22]]
dfA = pd.DataFrame(matriz_A)
dfA_transp = pd.DataFrame(dfA).T

# adição de matrizes
matriz_B = [[2,5],[3,7]]
dfB = pd.DataFrame(matriz_B)
matriz_C = [[1,3],[2,4]]
dfC = pd.DataFrame(matriz_C)
matriz_B_mais_C = dfB + dfC

# pág 22
# multiplicação de matrizes
matriz_D = [[2,5],[0,7],[4,3]]
dfD = pd.DataFrame(matriz_D)
matriz_E = [[4,2,1],[6,3,2]]
dfE = pd.DataFrame(matriz_E)
matriz_D_vezes_E = dfD.dot(dfE)
# a multiplicação de matrizes em ordem diferente gera resultados diferentes
matriz_E_vezes_D = dfE.dot(dfD)

# multiplicação de matriz por escalar
matriz_F = [[2,5],[3,7]]
dfF = pd.DataFrame(matriz_F)
matriz_tres_vezes_F = dfF * 3

# pág 23  - Solução de equações com 
# 4x1 + 10x2 = 38
# 10x1 + 30x2 = 110

matriz_G = [[4,10],[10,30]] # multiplicadores das incógnitas
dfG = pd.DataFrame(matriz_G)
matriz_H = [[38],[110]] # resultados
dfH = pd.DataFrame(matriz_H) # matriz dos resultados 
# Cálculo da matriz inversa
dfG_inv = pd.DataFrame(np.linalg.inv(dfG.values), dfG.columns, dfG.index)
# verificação da inversão da matriz
# a multiplicação da matriz inversa por ela mesma deve resultar na matriz identidade
matriz_identidade_G = dfG_inv.dot(dfG)
# cálculo das incógnitas x1 e x2
# mutiplicação da matriz inversa pela matriz dos resultados
matriz_resultado_incognitas = dfG_inv.dot(dfH)
# x1 = 2; x2 = 3

# pág 26 - exercício de cálculo de matriz de correlação

matriz_I = [[1,2,3],[2,3,4],[1,2,3],[5,4,3],[4,4,4]]
dfI = pd.DataFrame(matriz_I)
matriz_J = [[-0.889,-1,-0.727],[-0.333,0,1.091],[-0.889,-1,-0.727],[1.333,1,-0.727],[0.778,1,1.091]]
dfJ = pd.DataFrame(matriz_J) # matriz Z no livro
media_col_matriz_I = dfI.mean() # cálculo do desvio padrão de cada coluna da matriz_I
desvio_matriz_I = dfI.std() # cálculo do desvio padrão de cada coluna da matriz_I
# transposição da matriz_J
dfJ_transp = pd.DataFrame(dfJ).T
dfJtJ = dfJ_transp.dot(dfJ) # matriz V no livro
# para terminar o cálculo dos coeficientes de correlação
# basta multiplicar o escalar 1/(m-1) pela matriz V (no livro) 
m = 5 # número de observações
correl_I_J = (1/(m-1))*dfJtJ
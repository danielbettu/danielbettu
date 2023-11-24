# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:26:06 2023

@author: 02901717926
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from regressao import perform_linear_regression
from gardner_correl import perform_gardner_correl


# Carregar dados do arquivo de entrada em txt
df = pd.read_csv('C:/Python/eaton_Gov_Gp_Gc_Gf.txt', sep= "\t", header=None)

# Criando o dataframe 'dados' com os dados numéricos
dados = df.iloc[1:].apply(pd.to_numeric, errors='coerce')

# Criando o dataframe 'titulos' com os títulos das colunas
cabecalho = pd.DataFrame(df.iloc[0]).transpose().astype(str)


#####################################
# Cálculo da correlação de Gardner - estimativa da densidade da Fm

# Definição das variáveis
coef_a = 0.23 # coef de Gardner
coef_b = 0.25 # coef de Gardner

# Realizar a correlação de Gardner
dens_formacao = perform_gardner_correl(dados.iloc[:,1], coef_a, coef_b)

# Plotar resultado da densidade da fm
profundidade = dados.loc[:, 0]
# plt.plot(profundidade, dens_formacao)
# plt.show()

# Perguntar a profundidade de início da subcompactação
topo_subcomp = 3200#float(input("Qual é a profundidade de início da subcompactação? "))
print("A profundidade de início da subcompactação é:", topo_subcomp, ' m')

###################
# Gradiente de sobrecarga

# Definição das variáveis
dens_agua = 1.05 # g/cm3 
lamina_agua = 1000 # m
 
# cálculo de espessuras
# Calcular a diferença entre a linha atual e a próxima
espessura_camada = profundidade.diff()
# A primeira linha não tem uma linha anterior, então vamos preencher o NaN com o cálculo desejado
espessura_camada.iloc[0] = (profundidade.iloc[0]-lamina_agua)

# A última linha não tem uma próxima linha, então vamos preencher o NaN com a diferença para a profundidade anterior
espessura_camada.iloc[-1] = profundidade.iloc[-1] - profundidade.iloc[-2]

# Realizar o cálculo do gradiente de sobrecarga

tensao_camada = 1.422 * (espessura_camada * dens_formacao)
tensao_lamina_agua = 1.422 * (lamina_agua * dens_agua)
tensao_total = tensao_lamina_agua + tensao_camada.cumsum() 
gradiente_sobrecarga = tensao_total / (0.1704 * profundidade)

# Plotar 
#plt.plot(profundidade, gradiente_sobrecarga)  
#plt.xlabel('Profundidade (m)')
#plt.ylabel('Gradiente sobrecarga (lbf/gal)')

###################
# Cálculo da tendência linear de compactação
# Encontrar o índice da menor profundidade até 'topo_subcomp'
indices = (dados.iloc[:, 0] <= topo_subcomp)

# Realizar a regressão linear
x, y_pred, linear_regressor = perform_linear_regression(dados.loc[indices])

# Plotar dados e reta de tendência
# plt.plot(dados.iloc[:, 0], dados.iloc[:, 1], label='Dados originais')
# plt.plot(x, y_pred, color='red', label='Linha de tendência')  # Adicionar a linha de tendência ao gráfico
# plt.xlabel('Profundidade (m)')
# plt.ylabel('Vagarosidade (uS/ft)')
# plt.legend()  # Adicionar legenda ao gráfico
# plt.show()

# Imprimir na tela a equação da reta 
print("A equação da linha de tendência é: y = {:.2f}x + {:.2f}".format(linear_regressor.coef_[0][0], linear_regressor.intercept_[0]))

# Reshape your data
deltaT_medido = dados[1]

# Use the DataFrame for prediction
# Get the first coefficient
coef = linear_regressor.coef_[0][0]
deltaT_esperado = (coef * profundidade) + linear_regressor.intercept_


###################
# Estimativa de gradiente de pressão de poros - gradiente_poros
#gradiente_sobrecarga = gradiente_sobrecarga.to_frame()
gradiente_agua = tensao_lamina_agua/(0.1704 * lamina_agua)
gradiente_poros = gradiente_sobrecarga - ((gradiente_sobrecarga - gradiente_agua) * (deltaT_esperado/deltaT_medido)**2)


###################
# Estimativa de gradiente de pressão de colapso

C0 = 5000
phi = 55 #graus
tensao_hor_max = dados[2]
tensao_hor_min = dados[3]
pressao_poros_estimada = gradiente_poros * 0.1704 * profundidade
tan2phi = (math.tan((math.pi / 4) + (phi * (math.pi / 180))))**2
pressao_colapso_min = ((3 * tensao_hor_max) - tensao_hor_min - C0 + (pressao_poros_estimada * ( tan2phi - 1))) / tan2phi + 1
gradiente_colapso = pressao_colapso_min / (0.1704 * profundidade)

###################
# Estimativa de gradiente de fratura
#Gf = K (Gov – Gpp) + Gpp
coef_poisson = dados[4]
coef_K = coef_poisson / 1 - coef_poisson
gradiente_fratura = (coef_poisson * (gradiente_sobrecarga - gradiente_poros)) + gradiente_poros

# plt.plot(profundidade, gradiente_colapso)
# plt.plot(profundidade, gradiente_poros)
# plt.plot(profundidade, gradiente_sobrecarga)
# plt.plot(profundidade, gradiente_fratura)

###################
# Plotagem do gráfico
# Criação da figura e do eixo
fig, ax = plt.subplots()

# Plotagem das curvas
ax.plot(gradiente_colapso, profundidade, label='Gradiente de Colapso')
ax.plot(gradiente_poros, profundidade, label='Gradiente de Poros')
ax.plot(gradiente_sobrecarga, profundidade, label='Gradiente de Sobrecarga')
ax.plot(gradiente_fratura, profundidade, label='Gradiente de Fratura')

# Preenchimento da área entre as curvas gradiente_colapso e gradiente_sobrecarga
ax.fill_betweenx(profundidade, gradiente_colapso, gradiente_fratura, color='yellow', alpha=0.3)

# Inversão do eixo y para que a profundidade cresça para baixo
ax.invert_yaxis()

# Adição dos rótulos dos eixos
ax.set_xlabel('Gradientes (lbf/gal)')
ax.set_ylabel('Profundidade (m)')

# # Exibição da plotagem
# plt.show()

###################
# Plotagem de Círculo de Mohr

# Extração de dados
prof_interesse = float(input("Qual é a profundidade de interesse para geração do círculo de Mohr? "))
print("A profundidade de interesse para geração do círculo de Mohr é:", prof_interesse, ' m')

pressao_poros_interesse = pressao_poros_estimada.iloc[prof_interesse]
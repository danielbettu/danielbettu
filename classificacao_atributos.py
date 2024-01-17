#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:57:15 2024

@author: bettu
"""

import pandas as pd
import numpy as np

# Carregar dados do arquivo de atributos para análise de favorabilidade
df = pd.read_csv('https://raw.githubusercontent.com/danielbettu/danielbettu/main/atributos_entrada.csv', sep= ",", header=0)
sumario_df = df.describe().transpose() #quickly view summary statistcs for each column of data

# Carregar dados do arquivo de treinamento - ocorrência dos eventos
df_treino = pd.read_csv('https://raw.githubusercontent.com/danielbettu/danielbettu/main/treinamento_eventos.csv', sep= ",", header=0)


# criar dataframe com os valores dos quantis 0.33 e 0.66
def criar_df_quantil(df):
    quantil_33 = []
    quantil_66 = []
    for coluna in df.columns:
        quantil_33.append(df[coluna].quantile(0.33))
        quantil_66.append(df[coluna].quantile(0.66))
    data = {'Coluna': df.columns, 'Quantil_33': quantil_33, 'Quantil_66': quantil_66}
    df_quantil = pd.DataFrame(data)
    return df_quantil

df_quantil = criar_df_quantil(df)

# Função categorizar - para criar novo dataframe, categorizando o df original 
def categorizar(df):
    df_categorizado = pd.DataFrame()
    for coluna in df.columns:
        df_categorizado[coluna + '_baixo'] = np.where(df[coluna] < df[coluna].quantile(0.33), 1, 0)
        df_categorizado[coluna + '_moderado'] = np.where((df[coluna] >= df[coluna].quantile(0.33)) & (df[coluna] <= df[coluna].quantile(0.66)), 1, 0)
        df_categorizado[coluna + '_alto'] = np.where(df[coluna] > df[coluna].quantile(0.66), 1, 0)
    return df_categorizado

df_categorizado = categorizar(df)

# função para categorizar os atributos no arquivo de treinamento
def categorizar(df_treino, df_quantil):
    df_treino_categorizado = pd.DataFrame()
    for col in df_treino.columns:
        if col != df_treino.columns[-1]: # desconsiderando a última coluna do dataframe df_treino
            quantil_33 = df_quantil.loc[df_quantil['Coluna'] == col, 'Quantil_33'].values[0]
            quantil_66 = df_quantil.loc[df_quantil['Coluna'] == col, 'Quantil_66'].values[0]
            df_treino_categorizado[col + '_baixo'] = np.where(df_treino[col] < quantil_33, 1, 0)
            df_treino_categorizado[col + '_moderado'] = np.where((df_treino[col] >= quantil_33) & (df_treino[col] <= quantil_66), 1, 0)
            df_treino_categorizado[col + '_alto'] = np.where(df_treino[col] > quantil_66, 1, 0)
        else:
            df_treino_categorizado[col] = df_treino[col]
    return df_treino_categorizado

df_treino_categorizado = categorizar(df_treino, df_quantil)


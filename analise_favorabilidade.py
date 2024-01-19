#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 09:57:15 2024

@author: bettu
"""

import pandas as pd
import numpy as np

# Carregar dados do arquivo de atributos para análise de favorabilidade
#df = pd.read_csv('https://raw.githubusercontent.com/danielbettu/danielbettu/main/atributos_entrada.csv', sep= ",", header=0)
df = pd.read_csv('/home/bettu/Documents/Python/favorabilidade/dados_validacao8.csv', header=0)
sumario_df = df.describe().transpose() #quickly view summary statistcs for each column of data

# Carregar dados do arquivo de treinamento - ocorrência dos eventos
#df_treino = pd.read_csv('https://raw.githubusercontent.com/danielbettu/danielbettu/main/treinamento_eventos.csv', sep= ",", header=0)
df_treino = pd.read_csv('/home/bettu/Documents/Python/favorabilidade/dados_validacao_treino8.csv', header=0)

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
def categorizar(df_treino, df):
    df_treino_categorizado = pd.DataFrame()
    for col in df_treino.columns:
        if col != df_treino.columns[-1]: # desconsiderando a última coluna do dataframe df_treino
            quantil_33 = df[col].quantile(0.33)
            quantil_66 = df[col].quantile(0.66)
            df_treino_categorizado[col + '_baixo'] = np.where(df_treino[col] < quantil_33, 1, 0)
            df_treino_categorizado[col + '_moderado'] = np.where((df_treino[col] >= quantil_33) & (df_treino[col] <= quantil_66), 1, 0)
            df_treino_categorizado[col + '_alto'] = np.where(df_treino[col] > quantil_66, 1, 0)
        else:
            df_treino_categorizado[col] = df_treino[col]
    return df_treino_categorizado

df_treino_categorizado = categorizar(df_treino, df)

# Contagem de eventos e cálculo da probabilidade a priori
count = df_treino_categorizado[df_treino_categorizado.columns[-1]].value_counts()
# Atribuindo a contagem a variáveis
count_0 = count[0]
count_1 = count[1]
amostras_treino = len(df_treino_categorizado)
pE = count_1/amostras_treino
pnE = count_0/amostras_treino

# Início dos cálculos das probabilidades condicionais segundo a regra de Bayes

# cálculo das interseções evento vs atributos
def calculate_intersections(df):
    # Obtendo o nome da última coluna
    last_col = df.columns[-1]

    # Para armazenar os resultados
    results = {}

    # Iterando sobre cada coluna, exceto a última
    for col in df.columns[:-1]:
        # Calculando as probabilidades para cada combinação
        # para registro, a ordem das probabilidades indicadas
        # P_evento p_atributo ===> p_1_0 => evento=1 e atributo=0
        intersecao_E_A = len(df[(df[last_col] == 1) & (df[col] == 1)]) / len(df)
        intersecao_nE_A = len(df[(df[last_col] == 0) & (df[col] == 1)]) / len(df)
        intersecao_E_nA = len(df[(df[last_col] == 1) & (df[col] == 0)]) / len(df)
        intersecao_nE_nA = len(df[(df[last_col] == 0) & (df[col] == 0)]) / len(df)

        # Armazenando os resultados
        results[col] = {"intersecao_E_A": intersecao_E_A, "intersecao_nE_A": intersecao_nE_A, "intersecao_E_nA": intersecao_E_nA, "intersecao_nE_nA": intersecao_nE_nA}

    # Convertendo os resultados para um DataFrame para visualização fácil
    results_df = pd.DataFrame(results)

    return results_df

# Usando a função no DataFrame
results_df = calculate_intersections(df_treino_categorizado)

# Cálculo das probabilidades condicionais
probabilidades_condicionais = results_df.copy()

for col in results_df.columns:
    probabilidades_condicionais.loc['pE_A', col] = results_df.loc['intersecao_E_A', col] / pE
    probabilidades_condicionais.loc['pnE_A', col] = results_df.loc['intersecao_nE_A', col] / pnE
    probabilidades_condicionais.loc['pnE_nA', col] = results_df.loc['intersecao_nE_nA', col] / pnE
    probabilidades_condicionais.loc['pE_nA', col] = results_df.loc['intersecao_E_nA', col] / pE
    
for col in results_df.columns:
    probabilidades_condicionais.loc['fator_necessidade', col] = probabilidades_condicionais.loc['pnE_A', col] / probabilidades_condicionais.loc['pnE_nA', col]
    probabilidades_condicionais.loc['fator_suficiencia', col] = probabilidades_condicionais.loc['pE_A', col] / probabilidades_condicionais.loc['pnE_A', col]
    
# Fim do cálculo das probabilidades condicionais e fatores modificadores
# Início da análise de favorabilidade    
# a partir do dataframe df, é criado novo dataframe para predição
resultados_favorabilidade = df_categorizado.copy()

# Cópia do dataframe
resultados_favorabilidade = df_categorizado.copy()

# Adiciona uma nova coluna ao dataframe
resultados_favorabilidade['chance_posteriori'] = 1

# Itera sobre cada linha do dataframe
for index, row in resultados_favorabilidade.iterrows():
    # Define a chance prévia como 1. Esta é a probabilidade inicial antes de considerar os dados.
    chance_previa = 1

    # Itera sobre cada coluna do dataframe
    for col in resultados_favorabilidade.columns:
        # Exclui a nova coluna criada 'chance_posteriori' da iteração, pois ela não deve ser usada no cálculo da chance prévia.
        if col == 'chance_posteriori':
            continue

        # Calcula a chance prévia com base no valor em df_categorizado.
        # Se o valor na coluna correspondente de df_categorizado for 0, multiplica a chance prévia pela probabilidade condicional correspondente ao 'fator_necessidade'.
        # Se o valor na coluna correspondente de df_categorizado for 1, multiplica a chance prévia pela probabilidade condicional correspondente ao 'fator_suficiencia'.
        if df_categorizado.loc[index, col] == 0:
            chance_previa *= probabilidades_condicionais.loc['fator_necessidade', col]
        elif df_categorizado.loc[index, col] == 1:
            chance_previa *= probabilidades_condicionais.loc['fator_suficiencia', col]

    # Registra o valor da chance modificada na coluna 'chance_posteriori' do dataframe 'resultados_favorabilidade'.
    # Isso atualiza a chance prévia com as informações dos dados e armazena o resultado para referência futura.
    resultados_favorabilidade.loc[index, 'chance_posteriori'] = chance_previa

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:13:09 2023

@author: 02901717926
"""

import pandas as pd
import os

# Caminho para a pasta que contém os arquivos .txt
caminho_da_pasta = 'C:\Python\dados_teste'

# Lista para armazenar os dataframes
dataframes = []

# Percorre todos os arquivos na pasta especificada
for arquivo in os.listdir(caminho_da_pasta):
    # Verifica se o arquivo é um arquivo .txt
    if arquivo.endswith('.txt'):
        # Cria o caminho completo para o arquivo
        caminho_completo = os.path.join(caminho_da_pasta, arquivo)
        # Lê o arquivo .txt em um dataframe pandas
        df = pd.read_csv(caminho_completo, sep="\t")  # supondo que seja um arquivo tab-delimitado
        # Adiciona o dataframe à lista
        dataframes.append(df)

# Agora, 'dataframes' é uma lista de dataframes pandas, cada um correspondendo a um arquivo .txt

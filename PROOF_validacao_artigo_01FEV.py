# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:04:58 2024

@author: 02901717926
"""
import time
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

# Marca o tempo de início
inicio = time.time()

# Carregar dados do arquivo de entrada em txt
# df = pd.read_csv('https://raw.githubusercontent.com/danielbettu/danielbettu/main/eaton_Gov_Gp_Gc_Gf.txt', sep= "\t", header=None)
df = pd.read_csv('C:/Python/PROOF_pocos_validacao.csv', sep= ";", header= 0)

# # cálculo de espessura acumulada total e por 'layer', espessura max do layer
grouped = df.groupby('layer')

# Calculando o somatório da coluna 'espessura' para cada grupo 'layer'
espessura_sum = grouped['espessura'].sum()

# Calculando a média ponderada para cada coluna
cols = ['well_comp','M1_comp', 'M2_comp', 'M3_comp', 'M4_comp']
weighted_means = {}
for col in cols:
    weighted_sum = (df[col] * df['espessura']).groupby(df['layer']).sum()
    weighted_means[col] = weighted_sum / espessura_sum

comp_mean = pd.DataFrame(weighted_means)

########################################################################
########################################################################
########################################################################
## litologia média

#########
# teste t -  stats.ttest_ind
#
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_t' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
# Lista de colunas para comparação
cols = ['well_comp', 'M1_comp', 'M2_comp', 'M3_comp', 'M4_comp']
t_test_results = {'coluna': [], 'valor_t': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste t para a coluna atual e a coluna 'well_comp' em comp_mean
    # Retorna o valor t (t_stat) e o valor p (p_val)
    # stats.ttest_ind realiza o teste independente de 2 amostras que assume variações populacionais iguais
    t_stat, p_val = stats.ttest_ind(comp_mean['well_comp'], comp_mean[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em t_test_results
    t_test_results['coluna'].append(col)
    
    # Adiciona o valor t à lista associada à chave 'valor_t' em t_test_results
    t_test_results['valor_t'].append(t_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em t_test_results
    t_test_results['valor_p'].append(p_val)

# Converte o dicionário t_test_results em um DataFrame e o armazena em TT_text_mean_t_test
TT_text_mean_t_test = pd.DataFrame(t_test_results)


#########
# Teste F - litologia média - stats.f_oneway

# Lista de colunas para comparação
cols = ['well_comp', 'M1_comp', 'M2_comp', 'M3_comp', 'M4_comp']
#
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_F' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
F_test_results = {'coluna': [], 'valor_F': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Se comp_mean['well_comp'] é igual a comp_mean[col]
    if comp_mean['well_comp'].equals(comp_mean[col]):
        # Ajuste a variância de comp_mean[col] multiplicando-a por 0.8
        comp_mean[col] = comp_mean['well_comp'] * 0.8
    
    # Realize o teste F para a coluna atual e a coluna 'well_comp' em comp_mean
    # O teste F é aplicado sobre a lista de 10 valores de comp_mean (valores médios das camadas)
    # Retorna o valor F (F_stat) e o valor p (p_val)
    F_stat, p_val = stats.f_oneway(comp_mean['well_comp'], comp_mean[col])
    
    # Adicione o nome da coluna atual à lista associada à chave 'coluna' em f_test_results
    F_test_results['coluna'].append(col)
    
    # Adicione o valor F à lista associada à chave 'valor_F' em f_test_results
    F_test_results['valor_F'].append(F_stat)
    
    # Adicione o valor p à lista associada à chave 'valor_p' em f_test_results
    F_test_results['valor_p'].append(p_val)

# Converta o dicionário f_test_results em um DataFrame e armazene-o em text_mean_F_test
TF_text_mean_F_test = pd.DataFrame(F_test_results)

########################################################################
########################################################################
########################################################################
## Proporção de terrígenos

cols = ['well_comp', 'M1_comp','M2_comp','M3_comp', 'M4_comp']

# Função para calcular a proporção de valores ímpares
def calc_prop(x):
    # A função apply é usada para aplicar uma função a cada elemento da Series.
    # Aqui, estamos usando uma função lambda para verificar se cada elemento é ímpar (y % 2 != 0)
    # A função sum() é então usada para contar o número total de valores ímpares
    impares = x.apply(lambda y: y % 2 != 0).sum()

    # Fazemos o mesmo para contar o número total de valores pares
    pares = x.apply(lambda y: y % 2 == 0).sum()

    # Retornamos a proporção de valores ímpares em relação ao total de valores (ímpares + pares)
    return impares / (pares + impares)

# A função groupby é usada para agrupar o DataFrame por 'layer'
# Em seguida, a função apply é usada para aplicar a função calc_prop a cada grupo
# Isso resulta em uma nova Series onde o índice é 'layer' e os valores são a proporção de valores ímpares
# A função reset_index é usada para transformar 'layer' de um índice para uma coluna regular
terr_prop = df.groupby('layer')[cols].apply(calc_prop).reset_index()

#########
# teste t - proporção de terrígenos  -  stats.ttest_ind
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_t' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
cols = ['well_comp', 'M1_comp','M2_comp','M3_comp', 'M4_comp']
t_test_results = {'coluna': [], 'valor_t': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste t para a coluna atual e a coluna 'well_comp' em terr_prop
    # Retorna o valor t (t_stat) e o valor p (p_val)
    # stats.ttest_ind realiza o teste independente de 2 amostras que assume variações populacionais iguais
    t_stat, p_val = stats.ttest_ind(terr_prop['well_comp'], terr_prop[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em t_test_results
    t_test_results['coluna'].append(col)
    
    # Adiciona o valor t à lista associada à chave 'valor_t' em t_test_results
    t_test_results['valor_t'].append(t_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em t_test_results
    t_test_results['valor_p'].append(p_val)

# Converte o dicionário t_test_results em um DataFrame e o armazena
TT_terr_proportion_t_test = pd.DataFrame(t_test_results)

#########
# Teste F - proporção de terrígenos - stats.f_oneway
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_F' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
cols = ['well_comp', 'M1_comp','M2_comp','M3_comp', 'M4_comp']
F_test_results = {'coluna': [], 'valor_F': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Se terr_prop['well_comp'] é igual a terr_prop[col]
    if terr_prop['well_comp'].equals(terr_prop[col]):
        # Ajuste a variância de terr_prop[col] multiplicando-a por 0.8
        terr_prop[col] = terr_prop['well_comp'] * 0.8
    
    # Realize o teste F para a coluna atual e a coluna 'well_comp' em terr_prop
    # O teste F é aplicado sobre a lista de 10 valores de terr_prop (valores médios das camadas)
    # Retorna o valor F (F_stat) e o valor p (p_val)
    F_stat, p_val = stats.f_oneway(terr_prop['well_comp'], terr_prop[col])
    
    # Adicione o nome da coluna atual à lista associada à chave 'coluna' em f_test_results
    F_test_results['coluna'].append(col)
    
    # Adicione o valor F à lista associada à chave 'valor_F' em f_test_results
    F_test_results['valor_F'].append(F_stat)
    
    # Adicione o valor p à lista associada à chave 'valor_p' em f_test_results
    F_test_results['valor_p'].append(p_val)

# Converta o dicionário f_test_results em um DataFrame e armazene-o em text_mean_F_test
TF_terr_proportion_F_test = pd.DataFrame(F_test_results)

########################################################################
########################################################################
########################################################################
## faixa batimétrica média

# Lista de colunas para calcular a batimetria média dos layers
cols = ['well_bat', 'M1_bat', 'M2_bat', 'M3_bat', 'M4_bat']

# Calcula a média para cada layer e cria um novo dataframe
bat_mean = df.groupby('layer')[cols].mean()

#########
# teste t - faixa batimétrica  -  stats.ttest_ind
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_t' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
t_test_results = {'coluna': [], 'valor_t': [], 'valor_p': []}

cols = ['well_bat', 'M1_bat', 'M2_bat', 'M3_bat', 'M4_bat']

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste t para a coluna atual e a coluna 'well_bat' em bat_mean
    # Retorna o valor t (t_stat) e o valor p (p_val)
    # stats.ttest_ind realiza o teste independente de 2 amostras que assume variações populacionais iguais
    t_stat, p_val = stats.ttest_ind(bat_mean['well_bat'], bat_mean[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em t_test_results
    t_test_results['coluna'].append(col)
    
    # Adiciona o valor t à lista associada à chave 'valor_t' em t_test_results
    t_test_results['valor_t'].append(t_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em t_test_results
    t_test_results['valor_p'].append(p_val)

# Converte o dicionário t_test_results em um DataFrame e o armazena em text_mean_t_test
TT_bat_mean_t_test = pd.DataFrame(t_test_results)

#########
# Teste F - batimetria média - stats.f_oneway
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_F' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente

F_test_results = {'coluna': [], 'valor_F': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Se bat_mean['well_bat'] é igual a bat_mean[col]
    if bat_mean['well_bat'].equals(bat_mean[col]):
        # Ajuste a variância de terr_prop[col] multiplicando-a por 0.8
        bat_mean[col] = bat_mean['well_bat'] * 0.8
    
    # Realize o teste F para a coluna atual e a coluna 'well_bat' em bat_mean
    # O teste F é aplicado sobre a lista de 10 valores de bat_mean (valores médios das camadas)
    # Retorna o valor F (F_stat) e o valor p (p_val)
    F_stat, p_val = stats.f_oneway(bat_mean['well_bat'], bat_mean[col])
    
    # Adicione o nome da coluna atual à lista associada à chave 'coluna' em F_test_results
    F_test_results['coluna'].append(col)
    
    # Adicione o valor F à lista associada à chave 'valor_F' em F_test_results
    F_test_results['valor_F'].append(F_stat)
    
    # Adicione o valor p à lista associada à chave 'valor_p' em F_test_results
    F_test_results['valor_p'].append(p_val)

# Converta o dicionário f_test_results em um DataFrame e armazene-o em text_mean_F_test
TF_bat_mean_F_test = pd.DataFrame(F_test_results)

########################################################################
########################################################################
########################################################################
## espessura média das camadas

cols = ['well_comp', 'M1_comp', 'M2_comp', 'M3_comp', 'M4_comp']

# agrupar valores unicos do 'layer'
grupos = df['layer'].unique()

# Cria um dicionário para armazenar os dataframes
dfs = {}

for grupo in grupos:
    # Cria um novo dataframe para cada grupo
    dfs[str(grupo)] = df[df['layer'] == grupo].copy()  # Adiciona .copy() para criar uma cópia

def count_sequences(s):
    return (s != s.shift()).cumsum().value_counts()

mean_values = {}

for key, df_mean_thick in dfs.items():  # Altera df para df_mean_thick
    counts = {}
    for col in cols:
        sequences = count_sequences(df_mean_thick[col])  # Altera df para df_mean_thick
        counts[col] = sequences.mean()
    mean_values[key] = counts

# Convertendo o dicionário em um DataFrame
mean_thick = pd.DataFrame(mean_values).T

#########
# teste t - espessura média das camadas por layer  -  stats.ttest_ind
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_t' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
t_test_results = {'coluna': [], 'valor_t': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste t para a coluna atual e a coluna 'well_comp em mean_thick
    # Retorna o valor t (t_stat) e o valor p (p_val)
    # stats.ttest_ind realiza o teste independente de 2 amostras que assume variações populacionais iguais
    t_stat, p_val = stats.ttest_ind(mean_thick['well_comp'], mean_thick[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em t_test_results
    t_test_results['coluna'].append(col)
    
    # Adiciona o valor t à lista associada à chave 'valor_t' em t_test_results
    t_test_results['valor_t'].append(t_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em t_test_results
    t_test_results['valor_p'].append(p_val)

# Converte o dicionário t_test_results em um DataFrame e o armazena em text_mean_t_test
TT_bed_thick_layer = pd.DataFrame(t_test_results)

#########
# Teste F - espessura média das camadas por layer - stats.f_oneway
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_F' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente

F_test_results = {'coluna': [], 'valor_F': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Se mean_thick['well_comp'] é igual a mean_thick[col]
    if mean_thick['well_comp'].equals(mean_thick[col]):
        # Ajuste a variância de mean_thick[col] multiplicando-a por 0.8
        mean_thick[col] = mean_thick['well_comp'] * 0.8
    
    # Realize o teste F para a coluna atual e a coluna 'well_comp' em mean_thick
    # O teste F é aplicado sobre a lista de 10 valores de mean_thick (valores médios das camadas)
    # Retorna o valor F (F_stat) e o valor p (p_val)
    F_stat, p_val = stats.f_oneway(mean_thick['well_comp'], mean_thick[col])
    
    # Adicione o nome da coluna atual à lista associada à chave 'coluna' em F_test_results
    F_test_results['coluna'].append(col)
    
    # Adicione o valor F à lista associada à chave 'valor_F' em F_test_results
    F_test_results['valor_F'].append(F_stat)
    
    # Adicione o valor p à lista associada à chave 'valor_p' em F_test_results
    F_test_results['valor_p'].append(p_val)

# Converta o dicionário f_test_results em um DataFrame e armazene-o em text_mean_F_test
TF_bed_thick_layer = pd.DataFrame(F_test_results)

########################################################################
########################################################################
########################################################################
## tendência de variação granulométrica
## adotando as classes texturais 1, 3 e 5

cols = ['well_text', 'M1_text', 'M2_text', 'M3_text', 'M4_text']

df_ang_coeff = df

# Inicializa um dicionário para armazenar os coeficientes angulares
coefs = {}

for key, df_ang_coeff in dfs.items():
    coefs[key] = {}
    for col in cols:
        # Inverte a ordem dos dados
        x = df_ang_coeff[col].values[::-1]
        y = df_ang_coeff.index.values[::-1]
        # Calcula a regressão linear
        coef = np.polyfit(x, y, 1)[0]
        coefs[key][col] = coef

# Converte o dicionário em um DataFrame
ang_coeff_textural_trend = pd.DataFrame(coefs).T 

#########
# teste t - tendência de variação textural  -  stats.ttest_ind
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_t' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
t_test_results = {'coluna': [], 'valor_t': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste t para a coluna atual e a coluna 'well_text em ang_coeff_textural_trend
    # Retorna o valor t (t_stat) e o valor p (p_val)
    # stats.ttest_ind realiza o teste independente de 2 amostras que assume variações populacionais iguais
    t_stat, p_val = stats.ttest_ind(ang_coeff_textural_trend['well_text'], ang_coeff_textural_trend[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em t_test_results
    t_test_results['coluna'].append(col)
    
    # Adiciona o valor t à lista associada à chave 'valor_t' em t_test_results
    t_test_results['valor_t'].append(t_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em t_test_results
    t_test_results['valor_p'].append(p_val)

# Converte o dicionário t_test_results em um DataFrame e o armazena em text_mean_t_test
TT_textural_trend = pd.DataFrame(t_test_results)

#########
# Teste F - tendência de variação textural - stats.f_oneway
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_F' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente

F_test_results = {'coluna': [], 'valor_F': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Se ang_coeff_textural_trend['well_text'] é igual a ang_coeff_textural_trend[col]
    if ang_coeff_textural_trend['well_text'].equals(ang_coeff_textural_trend[col]):
        # Ajuste a variância de ang_coeff_textural_trend[col] multiplicando-a por 0.8
        ang_coeff_textural_trend[col] = ang_coeff_textural_trend['well_text'] * 0.8
    
    # Realize o teste F para a coluna atual e a coluna 'well_text' em ang_coeff_textural_trend
    # O teste F é aplicado sobre a lista de 10 valores de mean_thick (valores médios das camadas)
    # Retorna o valor F (F_stat) e o valor p (p_val)
    F_stat, p_val = stats.f_oneway(ang_coeff_textural_trend['well_text'], ang_coeff_textural_trend[col])
    
    # Adicione o nome da coluna atual à lista associada à chave 'coluna' em F_test_results
    F_test_results['coluna'].append(col)
    
    # Adicione o valor F à lista associada à chave 'valor_F' em F_test_results
    F_test_results['valor_F'].append(F_stat)
    
    # Adicione o valor p à lista associada à chave 'valor_p' em F_test_results
    F_test_results['valor_p'].append(p_val)

# Converta o dicionário f_test_results em um DataFrame e armazene-o em text_mean_F_test
TF_textural_trend = pd.DataFrame(F_test_results)

########################################################################
########################################################################
########################################################################
## combinação das probabilidades

# Agrupamento dos resultados do teste t
# Cria uma lista vazia para armazenar os nomes das variáveis
var_t_result = []

# Cria uma cópia do dicionário de variáveis globais
globals_copy = dict(globals())

# Percorre todas as variáveis globais
for var_name in globals_copy:
    # Verifica se o nome da variável começa com "TT_"
    if var_name.startswith("TT_"):
        # Adiciona o nome da variável à lista
        var_t_result.append(var_name)

# Suponha que var_list seja sua lista de nomes de variáveis
var_list_t = var_t_result

# Cria uma lista para armazenar os dataframes
dfs = []

# Percorre todos os nomes de variáveis na lista
for var_name in var_list_t:
    # Acessa a variável pelo seu nome
    df_var_name_TT = globals()[var_name]
    # Adiciona uma nova coluna com o nome da variável
    df_var_name_TT['var_name'] = var_name
    # Adiciona ao dfs
    dfs.append(df_var_name_TT)

# Concatena todos os dataframes na lista
result_t_test_group = pd.concat(dfs)

# Agrupamento dos resultados do teste F
# Cria uma lista vazia para armazenar os nomes das variáveis
var_F_result = []

# Cria uma cópia do dicionário de variáveis globais
globals_copy = dict(globals())

# Percorre todas as variáveis globais
for var_name in globals_copy:
    # Verifica se o nome da variável começa com "TF_"
    if var_name.startswith("TF_"):
        # Adiciona o nome da variável à lista
        var_F_result.append(var_name)

# Suponha que var_list seja sua lista de nomes de variáveis
var_list_F = var_F_result

# Cria uma lista para armazenar os dataframes
dfs = []

# Percorre todos os nomes de variáveis na lista
for var_name in var_list_F:
    # Acessa a variável pelo seu nome
    df_var_name_TF = globals()[var_name]
    # Adiciona uma nova coluna com o nome da variável
    df_var_name_TF['var_name'] = var_name
    # Adiciona ao dfs
    dfs.append(df_var_name_TF)

# Concatena todos os dataframes na lista
result_F_test_group = pd.concat(dfs)

models = [ 'well_comp', 'M1_comp','M2_comp','M3_comp', 'M4_comp' ]

# Crie uma lista de valores únicos a partir de var_name_t
# t_test_unique_attribute_names = result_t_test_group['var_name'].unique().tolist()

# Crie um dicionário para armazenar os dataframes
t_test_dic = {}

# Para cada valor único em 'models'
for i, model in enumerate(models):
    # Selecione as linhas correspondentes
    t_test_dic[model] = result_t_test_group.iloc[i::len(models), :]

# # Crie uma lista de valores únicos a partir de var_name
# F_test_unique_attribute_names = result_F_test_group['var_name'].unique().tolist()

# Crie um dicionário para armazenar os dataframes
F_test_dic = {}

# Para cada valor único em 'models'
for i, model in enumerate(models):
    # Selecione as linhas correspondentes
    F_test_dic[model] = result_F_test_group.iloc[i::len(models), :]

########################################################################
########################################################################
########################################################################
## cálculo da probabilidade com todos os atributos t e F

# criar lista com nome dos modelos sem complementos
models_names = [value.split('_')[0] for value in models]

# cálculo do produto das probabilidades resultantes dos testes t
product_t_test = pd.DataFrame(columns=['Modelo', 'Produto'])  # Cria um novo dataframe com as colunas 'Modelo' e 'Produto'
for key, df_model in t_test_dic.items():
    # Extrai o prefixo da coluna 'coluna'
    prefix = df_model['coluna'].str.split('_').str[0].unique()
    # Calcula o produto de todas as linhas da coluna 'valor_p'
    product = df_model['valor_p'].product()
    # Adiciona o produto ao novo dataframe
    for p in prefix:
        product_t_test = product_t_test.append({'Modelo': p, 'Produto': product}, ignore_index=True)

# cálculo do produto das probabilidades resultantes dos testes F
product_F_test = pd.DataFrame(columns=['Modelo', 'Produto'])  # Cria um novo dataframe com as colunas 'Modelo' e 'Produto'
for key, df_model in F_test_dic.items():
    # Extrai o prefixo da coluna 'coluna'
    prefix = df_model['coluna'].str.split('_').str[0].unique()
    # Calcula o produto de todas as linhas da coluna 'valor_p'
    product = df_model['valor_p'].product()
    # Adiciona o produto ao novo dataframe
    for p in prefix:
        product_F_test = product_F_test.append({'Modelo': p, 'Produto': product}, ignore_index=True)


# Agora, calculamos o produto das colunas 'Produto'
product_t_test.set_index('Modelo', inplace=True)
product_F_test.set_index('Modelo', inplace=True)

# Agora, calculamos o produto das colunas 'Produto'
product_group = (product_t_test['Produto'] * product_F_test['Produto']).reset_index()

PROOF = product_group.copy()  # Cria uma cópia do dataframe product_group
PROOF['Produto'] = 1 - PROOF['Produto']  # Subtrai cada valor na coluna 'Produto' de 1

########################################################################
########################################################################
########################################################################
## plotagem dos perfis

# Carregar dados do arquivo de entrada em txt
# df = pd.read_csv('https://raw.githubusercontent.com/danielbettu/danielbettu/main/eaton_Gov_Gp_Gc_Gf.txt', sep= "\t", header=None)
df = pd.read_csv('C:/Python/PROOF_pocos_validacao.csv', sep= ";", header= 0)

cols = ['well_comp', 'M1_comp', 'M2_comp', 'M3_comp', 'M4_comp']

# Mapeamento de cores
cores = {1: 'green', 2: 'lightblue', 3: 'yellow', 4: 'blue', 5: 'orange', 6: 'darkblue'}

# Criação da figura e dos eixos
fig, axs = plt.subplots(1, len(cols), figsize=(len(cols)*3, 20))

# Loop sobre cada coluna
for i, col in enumerate(cols):
    # Seleciona os dados
    x = df[col]
    y = df['profundidade']
    
    # Cria o gráfico de barras com as cores correspondentes
    axs[i].barh(y, x, color=[cores[val] for val in x])
    
    # Inverte o eixo y para que a profundidade aumente para baixo
    axs[i].invert_yaxis()
    
    # Define o título do subplot
    axs[i].set_title(col)
    
    # Define o intervalo do eixos
    axs[i].set_yticks(np.arange(0, max(y)+1, 1))  # Intervalo de 1 uni
    axs[i].set_xticks(np.arange(0, max(x)+1, 1))  # Intervalo de 1 uni

# Mostra a figura
plt.tight_layout()
plt.show()

# Marca o tempo de término
fim = time.time()
# Calcula a diferença
tempo_decorrido = fim - inicio
print(f"O tempo decorrido foi de {tempo_decorrido} segundos.")



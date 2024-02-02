# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:04:58 2024

@author: 02901717926
"""
import pandas as pd
from scipy import stats
import numpy as np


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

# Lista de colunas para comparação
cols = ['well_comp', 'M1_comp', 'M2_comp', 'M3_comp', 'M4_comp']

#########
# teste t -  stats.ttest_ind
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_t' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
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

# Converte o dicionário t_test_results em um DataFrame e o armazena em litho_mean_t_test
TT_litho_mean_t_test = pd.DataFrame(t_test_results)


#########
# Teste f - litologia média - stats.f_oneway
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_F' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
f_test_results = {'coluna': [], 'valor_F': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste F para a coluna atual e a coluna 'well_comp' em comp_mean
    # o teste F é aplicado sobre a lista de 10 valores de comp_mean (valores médios dos layers)
    # Retorna o valor F (F_stat) e o valor p (p_val)
    F_stat, p_val = stats.f_oneway(comp_mean['well_comp'], comp_mean[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em f_test_results
    f_test_results['coluna'].append(col)
    
    # Adiciona o valor F à lista associada à chave 'valor_F' em f_test_results
    f_test_results['valor_F'].append(F_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em f_test_results
    f_test_results['valor_p'].append(p_val)

# Converte o dicionário f_test_results em um DataFrame e o armazena em litho_mean_f_test
TF_litho_mean_f_test = pd.DataFrame(f_test_results)

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
# Teste f - proporção de terrígenos - stats.f_oneway
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_F' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
f_test_results = {'coluna': [], 'valor_F': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste F para a coluna atual e a coluna 'well_comp' em terr_prop
    # o teste F é aplicado sobre a lista de 10 valores de terr_prop
    # Retorna o valor F (F_stat) e o valor p (p_val)
    F_stat, p_val = stats.f_oneway(terr_prop['well_comp'], terr_prop[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em f_test_results
    f_test_results['coluna'].append(col)
    
    # Adiciona o valor F à lista associada à chave 'valor_F' em f_test_results
    f_test_results['valor_F'].append(F_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em f_test_results
    f_test_results['valor_p'].append(p_val)

# Converte o dicionário f_test_results em um DataFrame e o armazena em litho_mean_f_test
TF_terr_proportion_f_test = pd.DataFrame(f_test_results)

########################################################################
########################################################################
########################################################################
## faixa batimétrica média

# Lista de colunas para calcular a média
cols = ['well_bat', 'M1_bat', 'M2_bat', 'M3_bat', 'M4_bat']

# Calcula a média para cada layer e cria um novo dataframe
df_bath_mean = df.groupby('layer')[cols].mean()

#########
# teste t - faixa batimétrica  -  stats.ttest_ind
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_t' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
t_test_results = {'coluna': [], 'valor_t': [], 'valor_p': []}

cols = ['well_bat', 'M1_bat', 'M2_bat', 'M3_bat', 'M4_bat']

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste t para a coluna atual e a coluna 'well_bat' em df_bath_mean
    # Retorna o valor t (t_stat) e o valor p (p_val)
    # stats.ttest_ind realiza o teste independente de 2 amostras que assume variações populacionais iguais
    t_stat, p_val = stats.ttest_ind(df_bath_mean['well_bat'], df_bath_mean[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em t_test_results
    t_test_results['coluna'].append(col)
    
    # Adiciona o valor t à lista associada à chave 'valor_t' em t_test_results
    t_test_results['valor_t'].append(t_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em t_test_results
    t_test_results['valor_p'].append(p_val)

# Converte o dicionário t_test_results em um DataFrame e o armazena em litho_mean_t_test
TT_bat_mean_t_test = pd.DataFrame(t_test_results)

#########
# Teste f - batimetria média - stats.f_oneway
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_F' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
f_test_results = {'coluna': [], 'valor_F': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste F para a coluna atual e a coluna 'well_bat' em df_bath_mean
    # o teste F é aplicado sobre a lista de 10 valores de df_bath_mean
    # Retorna o valor F (F_stat) e o valor p (p_val)
    F_stat, p_val = stats.f_oneway(df_bath_mean['well_bat'], df_bath_mean[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em f_test_results
    f_test_results['coluna'].append(col)
    
    # Adiciona o valor F à lista associada à chave 'valor_F' em f_test_results
    f_test_results['valor_F'].append(F_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em f_test_results
    f_test_results['valor_p'].append(p_val)

# Converte o dicionário f_test_results em um DataFrame e o armazena em litho_mean_f_test
TF_bat_mean_f_test = pd.DataFrame(f_test_results)

########################################################################
########################################################################
########################################################################
## espessura média das camadas


# Sua lista de colunas
cols = ['well_comp', 'M1_comp', 'M2_comp', 'M3_comp', 'M4_comp']

# agrupar valores unicos do 'layer'
grupos = df['layer'].unique()

# Cria um dicionário para armazenar os dataframes
dfs = {}

for grupo in grupos:
    # Cria um novo dataframe para cada grupo
    dfs[ str(grupo)] = df[df['layer'] == grupo]

def count_sequences(s):
    return (s != s.shift()).cumsum().value_counts()

mean_values = {}

for key, df in dfs.items():
    counts = {}
    for col in cols:
        sequences = count_sequences(df[col])
        counts[col] = sequences.mean()
    mean_values[key] = counts

# Convertendo o dicionário em um DataFrame
mean_thick_df = pd.DataFrame(mean_values).T

#####
# mean_thick_df traz as espessuras 

#########
# teste t - espessura média das camadas por layer  -  stats.ttest_ind
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_t' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
t_test_results = {'coluna': [], 'valor_t': [], 'valor_p': []}

cols = ['well_comp', 'M1_comp', 'M2_comp', 'M3_comp', 'M4_comp']

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste t para a coluna atual e a coluna 'well_comp em mean_thick_df
    # Retorna o valor t (t_stat) e o valor p (p_val)
    # stats.ttest_ind realiza o teste independente de 2 amostras que assume variações populacionais iguais
    t_stat, p_val = stats.ttest_ind(mean_thick_df['well_comp'], mean_thick_df[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em t_test_results
    t_test_results['coluna'].append(col)
    
    # Adiciona o valor t à lista associada à chave 'valor_t' em t_test_results
    t_test_results['valor_t'].append(t_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em t_test_results
    t_test_results['valor_p'].append(p_val)

# Converte o dicionário t_test_results em um DataFrame e o armazena em litho_mean_t_test
TT_bed_thick_layer = pd.DataFrame(t_test_results)

#########
# Teste f - espessura média das camadas por layer - stats.f_oneway
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_F' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
f_test_results = {'coluna': [], 'valor_F': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste F para a coluna atual e a coluna 'well_comp' em df_bath_mean
    # o teste F é aplicado sobre a lista de 10 valores de df_bath_mean
    # Retorna o valor F (F_stat) e o valor p (p_val)
    F_stat, p_val = stats.f_oneway(mean_thick_df['well_comp'], mean_thick_df[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em f_test_results
    f_test_results['coluna'].append(col)
    
    # Adiciona o valor F à lista associada à chave 'valor_F' em f_test_results
    f_test_results['valor_F'].append(F_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em f_test_results
    f_test_results['valor_p'].append(p_val)

# Converte o dicionário f_test_results em um DataFrame e o armazena em litho_mean_f_test
TF_bed_thick_layer = pd.DataFrame(f_test_results)

########################################################################
########################################################################
########################################################################
## tendência de variação granulométrica
## adotando as classes texturais 1, 3 e 5
cols = ['well_lito', 'M1_lito', 'M2_lito', 'M3_lito', 'M4_lito']

# Inicializa um dicionário para armazenar os coeficientes angulares
coefs = {}

for key, df in dfs.items():
    coefs[key] = {}
    for col in cols:
        # Inverte a ordem dos dados
        x = df[col].values[::-1]
        y = df.index.values[::-1]
        # Calcula a regressão linear
        coef = np.polyfit(x, y, 1)[0]
        coefs[key][col] = coef

# Converte o dicionário em um DataFrame
coef_angular_tendencia_textural = pd.DataFrame(coefs).T

#########
# teste t - tendência de variação textural  -  stats.ttest_ind
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_t' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
t_test_results = {'coluna': [], 'valor_t': [], 'valor_p': []}

cols = ['well_lito', 'M1_lito', 'M2_lito', 'M3_lito', 'M4_lito']

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste t para a coluna atual e a coluna 'well_lito em coef_angular_tendencia_textural
    # Retorna o valor t (t_stat) e o valor p (p_val)
    # stats.ttest_ind realiza o teste independente de 2 amostras que assume variações populacionais iguais
    t_stat, p_val = stats.ttest_ind(coef_angular_tendencia_textural['well_lito'], coef_angular_tendencia_textural[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em t_test_results
    t_test_results['coluna'].append(col)
    
    # Adiciona o valor t à lista associada à chave 'valor_t' em t_test_results
    t_test_results['valor_t'].append(t_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em t_test_results
    t_test_results['valor_p'].append(p_val)

# Converte o dicionário t_test_results em um DataFrame e o armazena em litho_mean_t_test
TT_textural_trend = pd.DataFrame(t_test_results)

#########
# Teste f - tendência de variação textural - stats.f_oneway
# Inicializando um dicionário para armazenar os resultados
# As chaves do dicionário são 'coluna', 'valor_F' e 'valor_p'
# Cada chave corresponde a uma lista vazia que será preenchida posteriormente
f_test_results = {'coluna': [], 'valor_F': [], 'valor_p': []}

# Loop através de cada coluna em cols
for col in cols:
    # Realiza o teste F para a coluna atual e a coluna 'well_lito' em coef_angular_tendencia_textural
    # o teste F é aplicado sobre a lista de 10 valores 
    # Retorna o valor F (F_stat) e o valor p (p_val)
    F_stat, p_val = stats.f_oneway(coef_angular_tendencia_textural['well_lito'], coef_angular_tendencia_textural[col])
    
    # Adiciona o nome da coluna atual à lista associada à chave 'coluna' em f_test_results
    f_test_results['coluna'].append(col)
    
    # Adiciona o valor F à lista associada à chave 'valor_F' em f_test_results
    f_test_results['valor_F'].append(F_stat)
    
    # Adiciona o valor p à lista associada à chave 'valor_p' em f_test_results
    f_test_results['valor_p'].append(p_val)
    
# Converte o dicionário t_test_results em um DataFrame e o armazena em litho_mean_t_test
TF_textural_trend = pd.DataFrame(f_test_results)

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
var_list = var_t_result

# Cria uma lista para armazenar os dataframes
dfs = []

# Percorre todos os nomes de variáveis na lista
for var_name in var_list:
    # Acessa a variável pelo seu nome
    df = globals()[var_name]
    # Adiciona uma nova coluna com o nome da variável
    df['var_name'] = var_name
    # Adiciona ao dfs
    dfs.append(df)

# Concatena todos os dataframes na lista
result_t_test_group = pd.concat(dfs)

# Agrupamento dos resultados do teste F
# Cria uma lista vazia para armazenar os nomes das variáveis
var_t_result = []

# Cria uma cópia do dicionário de variáveis globais
globals_copy = dict(globals())

# Percorre todas as variáveis globais
for var_name in globals_copy:
    # Verifica se o nome da variável começa com "TF_"
    if var_name.startswith("TF_"):
        # Adiciona o nome da variável à lista
        var_t_result.append(var_name)

# Suponha que var_list seja sua lista de nomes de variáveis
var_list = var_t_result

# Cria uma lista para armazenar os dataframes
dfs = []

# Percorre todos os nomes de variáveis na lista
for var_name in var_list:
    # Acessa a variável pelo seu nome
    df = globals()[var_name]
    # Adiciona uma nova coluna com o nome da variável
    df['var_name'] = var_name
    # Adiciona ao dfs
    dfs.append(df)

# Concatena todos os dataframes na lista
result_F_test_group = pd.concat(dfs)


# # Agrupamento dos resultados do teste t
# # Inicializa um DataFrame vazio
# t_test_result_group = pd.DataFrame()

# # Cria uma cópia do dicionário de variáveis globais
# globals_copy = globals().copy()

# # Percorre todas as variáveis globais
# for var_name, var_value in globals_copy.items():
#     # Verifica se a variável é um dataframe e começa com "TT_"
#     if isinstance(var_value, pd.DataFrame) and var_name.startswith("TT_"):
#         for col in cols:
#             # Adiciona a coluna 'valor_p' ao dataframe
#             t_test_result_group[col] = var_value['valor_p']
            
# # Agrupamento dos resultados do teste F
# # Inicializa um dicionário para armazenar os dados
# data = {}

# # Cria uma cópia do dicionário de variáveis globais
# globals_copy = globals().copy()

# # Percorre todas as variáveis globais
# for var_name, var_value in globals_copy.items():
#     # Verifica se a variável é um dataframe e começa com "TF_"
#     if isinstance(var_value, pd.DataFrame) and var_name.startswith("TF_"):
#         for col in cols:
#             # Adiciona a coluna 'valor_p' ao dicionário
#             data[col] = var_value['valor_p']

# # Converte o dicionário em um DataFrame
# f_result_group = pd.DataFrame(data)

print(   'FIM!')

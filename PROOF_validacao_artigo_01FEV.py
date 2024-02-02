# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:04:58 2024

@author: 02901717926
"""
import pandas as pd
from scipy import stats

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
## 
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

########################################################################
########################################################################
########################################################################
## Classe Litológica Textural Média


# # cálculo de espessura acumulada total e por 'layer', espessura max do layer
grouped = df.groupby('layer')
thick_sum_layer = grouped['espessura'].sum() # Calculando o somatório da coluna 'espessura' para cada grupo 'layer'

grouped = df.groupby('layer')
cols = ['well_text','M1_text', 'M2_text', 'M3_text', 'M4_text']

# Função para encontrar o valor mais comum em cada coluna
def most_frequent_and_max(s):
    counts = s.value_counts()
    max_count = counts.max()
    most_frequent_values = counts[counts == max_count].index
    return most_frequent_values.max()

lit_dom = df.groupby('layer')[cols].agg(most_frequent_and_max).reset_index() # litologia dominante em cada layer

dict_mean_lit_dom = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a média ponderada usando os valores em thick_sum_layer como pesos
    weighted_avg = np.average(lit_dom[col], weights=np.sqrt(thick_sum_layer))
       
    # Crie um novo dataframe com a média ponderada
    dict_mean_lit_dom["mean_lit_dom_" + col] = pd.DataFrame([weighted_avg], columns=[col])
    
dict_var_lit_dom = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a variância
    variance = np.var(lit_dom[col])
       
    # Crie um novo dataframe com a média ponderada
    dict_var_lit_dom["var_lit_dom_" + col] = pd.DataFrame([variance], columns=[col])
    
########################################################################
########################################################################
########################################################################
## Proporção de Terrígenos média do poço 

cols = ['well_comp', 'M1_comp','M2_comp','M3_comp', 'M4_comp']

# Função para calcular a proporção de valores ímpares (terrígenos = ímpar; carbonatos = par)
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

dict_mean_terr_prop = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a média ponderada usando os valores em thick_sum_layer como pesos
    weighted_avg = np.average(terr_prop[col], weights=np.sqrt(thick_sum_layer))
       
    # Crie um novo dataframe com a média ponderada
    dict_mean_terr_prop["mean_terr_prop_" + col] = pd.DataFrame([weighted_avg], columns=[col])

dict_var_terr_prop = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a variância
    variance = np.var(terr_prop[col])
       
    # Crie um novo dataframe com a média ponderada
    dict_var_terr_prop["var_terr_prop_" + col] = pd.DataFrame([variance], columns=[col])
    
########################################################################
########################################################################
########################################################################
## faixa batimétrica média

# Lista de colunas para calcular a batimetria média dos layers
cols = ['well_bat', 'M1_bat', 'M2_bat', 'M3_bat', 'M4_bat']

# Calcula a média para cada layer e cria um novo dataframe
bat_mean = df.groupby('layer')[cols].mean()

dict_mean_bat = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a média 
    average = np.average(bat_mean[col])
       
    # Crie um novo dataframe com a média 
    dict_mean_bat["mean_bat_" + col] = pd.DataFrame([average], columns=[col])
    
dict_var_bat = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a variância
    variance = np.var(bat_mean[col])
       
    # Crie um novo dataframe com a média ponderada
    dict_var_bat["var_bat_" + col] = pd.DataFrame([variance], columns=[col])

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
# s.shift(): Este método desloca os índices de uma Série pandas 's' por um número especificado de períodos, que é 1 por padrão. 
# Isso significa que cada elemento em 's' é movido para baixo por uma posição, e o primeiro elemento é substituído por NaN.

# s != s.shift(): Esta linha de código compara cada elemento em 's' com o elemento abaixo dele (devido ao deslocamento). 
# Se os dois elementos forem diferentes, o resultado é True; se forem iguais, o resultado é False. 
# Isso retorna uma nova Série de valores booleanos do mesmo tamanho que 's'.

# (s != s.shift()).cumsum(): O método cumsum() é aplicado à Série booleana, que retorna a soma cumulativa dos valores. 
# Em outras palavras, ele adiciona os valores True (considerados como 1) à medida que avança pela Série.

# (s != s.shift()).cumsum().value_counts(): Por fim, o método value_counts() conta a ocorrência de cada valor único na Série cumsum. 
# Isso retorna uma nova Série onde o índice é o valor único da Série original e o valor é a contagem de ocorrências desse valor.
# Em resumo, este código está contando a frequência de sequências consecutivas de valores idênticos na Série 's'.

mean_values = {}

for key, df_mean_thick in dfs.items():  # Altera df para df_mean_thick
    counts = {}
    for col in cols:
        sequences = count_sequences(df_mean_thick[col])  # Altera df para df_mean_thick
        counts[col] = sequences.mean()
    mean_values[key] = counts

# Convertendo o dicionário em um DataFrame
mean_thick = pd.DataFrame(mean_values).T
sqrt_mean_thick = mean_thick.copy()
sqrt_mean_thick = sqrt_mean_thick.applymap(np.sqrt) # Atualiza todas as colunas do dataframe sqrt_mean_thick para que contenham o valor da raiz quadrada do valor original

dict_mean_sqrt_thick = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a média
    average = np.average(sqrt_mean_thick[col])
       
    # Crie um novo dataframe com a média 
    dict_mean_sqrt_thick["mean_layer_thick_" + col] = pd.DataFrame([average], columns=[col])

dict_var_sqrt_thick = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a variância
    variance = np.var(sqrt_mean_thick[col])
       
    # Crie um novo dataframe com a média ponderada
    dict_var_sqrt_thick["var_sqrt_thick_" + col] = pd.DataFrame([variance], columns=[col])

#######################################################################
#######################################################################
#######################################################################
# tendência de variação granulométrica vertical
# adotando as classes texturais 1, 3 e 5
# para fins de desenvolvimento e apresentação da PROOF em artigo
# a tendência de variação textural será calculada para o poço como um todo
# e não por sequência (isso será feito no artigo)
cols = ['well_text', 'M1_text', 'M2_text', 'M3_text', 'M4_text']

df_ang_coeff = df.copy()

# Primeiro, vamos definir os valores que consideramos 'grossos' e 'finos'
grossos_vals = [3, 5]
finos_vals = [1]

# Agora, vamos criar um dicionário para armazenar os resultados temporários
coarse_prop_dict = {}

# Iteramos sobre cada 'layer' único no DataFrame original
for layer in df_ang_coeff['layer'].unique():
    # Criamos um subconjunto do DataFrame original que contém apenas as linhas para o 'layer' atual
    df_subset = df_ang_coeff[df_ang_coeff['layer'] == layer]
    
    # Inicializamos um dicionário para armazenar os resultados para o 'layer' atual
    coarse_prop_dict[layer] = {}
    
    # Iteramos sobre cada coluna especificada em 'cols'
    for col in cols:
        # Calculamos a quantidade de 'grossos' e 'finos' para o 'layer' e 'col' atuais
        grossos = df_subset[col].isin(grossos_vals).sum()
        finos = df_subset[col].isin(finos_vals).sum()
        
        # Calculamos a proporção de 'grossos' para o total de 'grossos' e 'finos'
        coarse_prop = grossos / (grossos + finos)
        
        # Adicionamos o resultado ao nosso dicionário temporário
        coarse_prop_dict[layer][col] = coarse_prop

# Convertemos o dicionário de resultados em um DataFrame
df_coarse_prop = pd.DataFrame(coarse_prop_dict).T

# Inicializa um dicionário para armazenar os coeficientes angulares
coefs = {}

# Itera sobre cada coluna especificada em 'cols'
for col in cols:
    # Inverte a ordem dos dados na coluna atual
    x = df_coarse_prop[col].values[::-1]
    # Inverte a ordem dos índices do DataFrame atual
    y = df_coarse_prop.index.values[::-1]
    # Calcula a regressão linear dos dados invertidos e extrai o coeficiente angular
    coef = np.polyfit(x, y, 1)[0]
    # Armazena o coeficiente angular no dicionário
    coefs[col] = coef

# Inicializa um dicionário para armazenar os DataFrames
dict_text_trend = {}

# Itera sobre cada coluna especificada em 'cols'
for col in cols:
    # Cria um DataFrame para a coluna atual e armazena no dicionário
    dict_text_trend[col] = pd.DataFrame({col: coefs[col]}, index=[0])
    
# # ########################################################################
# # ########################################################################
# # ########################################################################
# # ## combinação das probabilidades

# # # Agrupamento dos resultados do teste t
# # # Cria uma lista vazia para armazenar os nomes das variáveis
# # var_t_result = []

# # # Cria uma cópia do dicionário de variáveis globais
# # globals_copy = dict(globals())

# # # Percorre todas as variáveis globais
# # for var_name in globals_copy:
# #     # Verifica se o nome da variável começa com "TT_"
# #     if var_name.startswith("TT_"):
# #         # Adiciona o nome da variável à lista
# #         var_t_result.append(var_name)

# # # Suponha que var_list seja sua lista de nomes de variáveis
# # var_list_t = var_t_result

# # # Cria uma lista para armazenar os dataframes
# # dfs = []

# # # Percorre todos os nomes de variáveis na lista
# # for var_name in var_list_t:
# #     # Acessa a variável pelo seu nome
# #     df_var_name_TT = globals()[var_name]
# #     # Adiciona uma nova coluna com o nome da variável
# #     df_var_name_TT['var_name'] = var_name
# #     # Adiciona ao dfs
# #     dfs.append(df_var_name_TT)

# # # Concatena todos os dataframes na lista
# # result_t_test_group = pd.concat(dfs)

# # # Agrupamento dos resultados do teste F
# # # Cria uma lista vazia para armazenar os nomes das variáveis
# # var_F_result = []

# # # Cria uma cópia do dicionário de variáveis globais
# # globals_copy = dict(globals())

# # # Percorre todas as variáveis globais
# # for var_name in globals_copy:
# #     # Verifica se o nome da variável começa com "TF_"
# #     if var_name.startswith("TF_"):
# #         # Adiciona o nome da variável à lista
# #         var_F_result.append(var_name)

# # # Suponha que var_list seja sua lista de nomes de variáveis
# # var_list_F = var_F_result

# # # Cria uma lista para armazenar os dataframes
# # dfs = []

# # # Percorre todos os nomes de variáveis na lista
# # for var_name in var_list_F:
# #     # Acessa a variável pelo seu nome
# #     df_var_name_TF = globals()[var_name]
# #     # Adiciona uma nova coluna com o nome da variável
# #     df_var_name_TF['var_name'] = var_name
# #     # Adiciona ao dfs
# #     dfs.append(df_var_name_TF)

# # # Concatena todos os dataframes na lista
# # result_F_test_group = pd.concat(dfs)

# # models = [ 'well_comp', 'M1_comp','M2_comp','M3_comp', 'M4_comp' ]

# # # Crie uma lista de valores únicos a partir de var_name_t
# # # t_test_unique_attribute_names = result_t_test_group['var_name'].unique().tolist()

# # # Crie um dicionário para armazenar os dataframes
# # t_test_dic = {}

# # # Para cada valor único em 'models'
# # for i, model in enumerate(models):
# #     # Selecione as linhas correspondentes
# #     t_test_dic[model] = result_t_test_group.iloc[i::len(models), :]

# # # # Crie uma lista de valores únicos a partir de var_name
# # # F_test_unique_attribute_names = result_F_test_group['var_name'].unique().tolist()

# # # Crie um dicionário para armazenar os dataframes
# # F_test_dic = {}

# # # Para cada valor único em 'models'
# # for i, model in enumerate(models):
# #     # Selecione as linhas correspondentes
# #     F_test_dic[model] = result_F_test_group.iloc[i::len(models), :]

# # ########################################################################
# # ########################################################################
# # ########################################################################
# # ## cálculo da probabilidade com todos os atributos t e F

# # # pré-cálculo do produto das probabilidades
# # # para evitar valores tendendo a zero ou infinito, são impostos limites mínimo 0.001 e máximo 0.9 (conforme arquivo PROOF_Workflow_Boing.docx)

# # # Aplicando os limites em cada DataFrame no dicionário t_test_dic
# # for key in t_test_dic.keys():
# #     if 'valor_p' in t_test_dic[key].columns:
# #         t_test_dic[key]['valor_p'] = t_test_dic[key]['valor_p'].clip(lower=0.001, upper=0.9)
        
# # # Aplicando os limites em cada DataFrame no dicionário F_test_dic
# # for key in F_test_dic.keys():
# #     if 'valor_p' in F_test_dic[key].columns:
# #         F_test_dic[key]['valor_p'] = F_test_dic[key]['valor_p'].clip(lower=0.001, upper=0.9)
        
# # # criar lista com nome dos modelos sem complementos
# # models_names = [value.split('_')[0] for value in models]

# # # cálculo do produto das probabilidades resultantes dos testes t
# # product_t_test = pd.DataFrame(columns=['Modelo', 'Produto'])  # Cria um novo dataframe com as colunas 'Modelo' e 'Produto'
# # for key, df_model in t_test_dic.items():
# #     # Extrai o prefixo da coluna 'coluna'
# #     prefix = df_model['coluna'].str.split('_').str[0].unique()
# #     # Calcula o produto de todas as linhas da coluna 'valor_p'
# #     product = df_model['valor_p'].product()
# #     # Adiciona o produto ao novo dataframe
# #     for p in prefix:
# #         product_t_test = product_t_test.append({'Modelo': p, 'Produto': product}, ignore_index=True)

# # # cálculo do produto das probabilidades resultantes dos testes F
# # product_F_test = pd.DataFrame(columns=['Modelo', 'Produto'])  # Cria um novo dataframe com as colunas 'Modelo' e 'Produto'
# # for key, df_model in F_test_dic.items():
# #     # Extrai o prefixo da coluna 'coluna'
# #     prefix = df_model['coluna'].str.split('_').str[0].unique()
# #     # Calcula o produto de todas as linhas da coluna 'valor_p'
# #     product = df_model['valor_p'].product()
# #     # Adiciona o produto ao novo dataframe
# #     for p in prefix:
# #         product_F_test = product_F_test.append({'Modelo': p, 'Produto': product}, ignore_index=True)

# # # Agora, calculamos o produto das colunas 'Produto'
# # product_t_test.set_index('Modelo', inplace=True)
# # product_F_test.set_index('Modelo', inplace=True)

# # # Agora, calculamos o produto das colunas 'Produto'
# # product_group = (product_t_test['Produto'] * product_F_test['Produto']).reset_index()

# # PROOF = product_group.copy()  # Cria uma cópia do dataframe product_group
# # PROOF.rename(columns={'Produto': 'PROOF'}, inplace=True) # Renomeia a coluna 'Produto' para 'PROOF'
# # PROOF['PROOF'] = 1 - PROOF['PROOF'] # Atualiza a coluna 'PROOF' para ser 1 - 'PROOF'

# # # PROOF = PROOF_pre.copy()
# # # PROOF = 1 - product_group['Produto']

# # ########################################################################
# # ########################################################################
# # ########################################################################
# # ## plotagem dos perfis

# # cols = ['well_comp', 'M1_comp', 'M2_comp', 'M3_comp', 'M4_comp']

# # # Mapeamento de cores
# # cores = {1: 'green', 2: 'lightblue', 3: 'yellow', 4: 'blue', 5: 'orange', 6: 'darkblue'}

# # # Criação da figura e dos eixos
# # fig, axs = plt.subplots(1, len(cols), figsize=(len(cols)*3, 20))

# # # Loop sobre cada coluna
# # for i, col in enumerate(cols):
# #     # Seleciona os dados
# #     x = df[col]
# #     y = df['profundidade']
    
# #     # Cria o gráfico de barras com as cores correspondentes
# #     axs[i].barh(y, x, color=[cores[val] for val in x])
    
# #     # Inverte o eixo y para que a profundidade aumente para baixo
# #     axs[i].invert_yaxis()
    
# #     # Define o título do subplot
# #     axs[i].set_title(col)
    
# #     # Define o intervalo do eixos
# #     axs[i].set_yticks(np.arange(0, max(y)+1, 1))  # Intervalo de 1 uni
# #     axs[i].set_xticks(np.arange(0, max(x)+1, 1))  # Intervalo de 1 uni

# # # Mostra a figura
# # plt.tight_layout()
# # plt.show()

# # # Marca o tempo de término
# # fim = time.time()
# # # Calcula a diferença
# # tempo_decorrido = fim - inicio
# # print(f"O tempo decorrido foi de {tempo_decorrido} segundos.")



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

dict_mean_litho_dom = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a média ponderada usando os valores em thick_sum_layer como pesos
    weighted_avg = np.average(lit_dom[col], weights=np.sqrt(thick_sum_layer))
       
    # Crie um novo dataframe com a média ponderada
    dict_mean_litho_dom[col] = pd.DataFrame([weighted_avg], columns=[col])
    
dict_var_litho_dom = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a variância
    variance = np.var(lit_dom[col])
       
    # Crie um novo dataframe com a média ponderada
    dict_var_litho_dom[col] = pd.DataFrame([variance], columns=[col])
    
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
    dict_mean_terr_prop[col] = pd.DataFrame([weighted_avg], columns=[col])

dict_var_terr_prop = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a variância
    variance = np.var(terr_prop[col])
       
    # Crie um novo dataframe com a média ponderada
    dict_var_terr_prop[col] = pd.DataFrame([variance], columns=[col])
    
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
    dict_mean_bat[col] = pd.DataFrame([average], columns=[col])
    
dict_var_bat = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a variância
    variance = np.var(bat_mean[col])
       
    # Crie um novo dataframe com a média ponderada
    dict_var_bat[col] = pd.DataFrame([variance], columns=[col])

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
    dict_mean_sqrt_thick[col] = pd.DataFrame([average], columns=[col])

dict_var_sqrt_thick = {} # Crie um dicionário para armazenar os dataframes

for col in cols:
    # Calcule a variância
    variance = np.var(sqrt_mean_thick[col])
       
    # Crie um novo dataframe com a média ponderada
    dict_var_sqrt_thick[col] = pd.DataFrame([variance], columns=[col])

#######################################################################
#######################################################################
#######################################################################
# tendência de variação granulométrica vertical
# adotando as classes texturais 1, 3 e 5
# para fins de desenvolvimento e apresentação da PROOF em artigo
# a tendência de variação textural será calculada para o poço como um todo
# e não por sequência (isso será feito no artigo)

cols = ['well_text', 'M1_text', 'M2_text', 'M3_text', 'M4_text']

df_ang_coef = df.copy()

# Inicializa o dicionário final
ang_coef_dict = {}

# cálculod do coeficiente angular - Loop através de cada coluna/modelo
for col in cols:
    # Cria um DataFrame com os valores
    df_b_coeff = pd.DataFrame({'y': df_ang_coef[col], 'x': range(len(df_ang_coef[col]))})
    
    # Calcula a covariância entre x e y
    covariance = df_b_coeff['x'].cov(df_b_coeff['y'])
    
    # Calcula a variância de x
    variance = df_b_coeff['x'].var()
    
    # O coeficiente angular é a covariância dividida pela variância
    angular_coeff = covariance / variance
    
    # Adiciona ao dicionário final
    ang_coef_dict[col] = angular_coeff
    
# cáclulo da variância de transições

num_rows = len(df_ang_coef.index)  # Número de linhas em df_ang_coef

dict_mean_vert_trend = {}

for key, value in ang_coef_dict.items():
    new_value = value * (num_rows - 1)  # Aplica a fórmula
    dict_mean_vert_trend[key] = new_value  # Adiciona o novo valor ao dicionário


# # ########################################################################
# # ########################################################################
# # ########################################################################
# # ## Variabilidade litológica - frequência de transição de litologia

# Define as colunas que representam os modelos
cols = ['well_comp', 'M1_comp', 'M2_comp', 'M3_comp', 'M4_comp']

# Cria uma cópia do DataFrame original para calcular as mudanças
df_litho_change = df.copy()

# cálculo da variabilidade média 

# Inicializa o dicionário final
dict_mean_litho_change = {}

# # Inicializa o DataFrame intermediário
# verificacao_numero_transicoes = pd.DataFrame()

# Loop através de cada coluna/modelo
for col in cols:
    # Calcula a diferença entre cada valor e o valor anterior
    diff = df_litho_change[col].diff()
    
    # Conta o número de mudanças (ignorando a primeira linha que será NaN devido ao cálculo da diferença "-1")
    change_count = (diff != 0).sum()-1
    
    # # Adiciona a contagem de mudanças ao DataFrame intermediário
    # verificacao_numero_transicoes[col] = [change_count]
    
    # Calcula o valor final como o número de mudanças dividido pelo número de camadas -1
    final_value = change_count / (len(df_litho_change)-1)
    
    # Cria um DataFrame para este modelo
    df_litho_change_value = pd.DataFrame([final_value], columns=['final_value'])
    
    # Adiciona ao dicionário final
    dict_mean_litho_change[col] = df_litho_change_value

# cáclulo da variância de transições


dict_var_litho_change = {}

for key, df_temp in dict_mean_litho_change.items():
    new_df = df_temp.copy()  # Cria uma cópia do dataframe
    new_df['final_value'] = new_df['final_value'].apply(lambda x: x * (1 - x))  # Aplica a fórmula
    dict_var_litho_change[key] = new_df  # Adiciona o novo dataframe ao dicionário

   
# ########################################################################
# ########################################################################
# ########################################################################
# ## agrupamento dos dicionários

# limpando os nomes das linhas nos dicionários
global_dict_copy = dict(globals())  # cria uma cópia do dicionário global
for name_dict, value_dict in global_dict_copy.items():
    if name_dict.startswith("dict_"):
        new_value_dict = {}
        for line_name, line_value in value_dict.items():
            new_line_name = line_name.split("_")[0]  # mantém apenas caracteres iniciais, corta a partir de "_"
            new_value_dict[new_line_name] = line_value
        globals()[name_dict] = new_value_dict

# transformando os dataframes dos dicts em floating point
global_dict_copy = dict(globals())  # cria uma cópia do dicionário global
for name_dict, value_dict in global_dict_copy.items():
    if name_dict.startswith("dict_"):
        for model_name, model_value in value_dict.items():
            if isinstance(model_value, pd.DataFrame) and model_value.shape == (1, 1):
                # Converte o dataframe 1x1 para float64
                value_dict[model_name] = model_value.iloc[0, 0]

# unificando os dicionários                
global_dict_copy = dict(globals())

# Cria um dicionário para armazenar os DataFrames
model_dfs = {}

for name_dict, value_dict in global_dict_copy.items():
    if name_dict.startswith("dict_"):
        # Extrai o nome do atributo do nome do dicionário
        atributo_name = name_dict.replace("dict_", "")
        for model_name, model_value in value_dict.items():
            # Se o modelo já tem um DataFrame, adiciona as novas linhas
            if model_name in model_dfs:
                if isinstance(model_value, pd.DataFrame):
                    model_value['atributo'] = atributo_name  # Adiciona uma nova coluna com o nome do atributo
                    model_dfs[model_name] = pd.concat([model_dfs[model_name], model_value], ignore_index=True)
                else:
                    temp_df = pd.DataFrame([model_value])
                    temp_df['atributo'] = atributo_name  # Adiciona uma nova coluna com o nome do atributo
                    model_dfs[model_name] = pd.concat([model_dfs[model_name], temp_df], ignore_index=True)
            # Se o modelo não tem um DataFrame, cria um novo
            else:
                if isinstance(model_value, pd.DataFrame):
                    model_value['atributo'] = atributo_name  # Adiciona uma nova coluna com o nome do atributo
                    model_dfs[model_name] = model_value
                else:
                    temp_df = pd.DataFrame([model_value])
                    temp_df['atributo'] = atributo_name  # Adiciona uma nova coluna com o nome do atributo
                    model_dfs[model_name] = temp_df

# Renomeia os DataFrames
for model_name, df_temp2 in model_dfs.items():
    globals()["atributos_" + model_name] = df_temp2

# # ########################################################################
# # ########################################################################
# # ########################################################################
# # ## cálculo da probabilidade com todos os atributos t e F

# # separação dos atributos media e variância
# # Cria uma cópia do dicionário de nomes globais
# globals_copy = dict(globals())
# # Percorre todos os nomes na cópia
# for name, value in globals_copy.items():
#     # Verifica se o valor é um dataframe, o nome começa com 'atributos_' e 'atributo' é uma coluna do dataframe
#     if isinstance(value, pd.DataFrame) and name.startswith('atributos_') and 'atributo' in value.columns:
#         # Cria o sub-dataframe 'attrib_medias' com as linhas onde 'atributo' começa com 'mean_'
#         globals()[name + '_mean'] = value[value['atributo'].str.startswith('mean_')]
#         # Cria o sub-dataframe 'attrib_var' com as linhas onde 'atributo' começa com 'var_'
#         globals()[name + '_var'] = value[value['atributo'].str.startswith('var_')]


# Copia do dicionário global
globals_copy = globals().copy()

# Lista de todos os seus dataframes que começam com 'atributos_', exceto 'atributos_well'
dataframes = {nome: dataframe for nome, dataframe in globals_copy.items() if nome.startswith('atributos_') and nome != 'atributos_well'}

resultados = []

for nome, dataframe in dataframes.items():
    # Obter o nome do modelo removendo 'atributos_' do nome da variável
    modelo = nome.replace('atributos_', '')
    
    for atributo in dataframe['atributo']:
        if atributo.startswith('mean_'):
            t_resultado = stats.ttest_ind(globals_copy['atributos_well'][globals_copy['atributos_well']['atributo'] == atributo].iloc[:, 0], dataframe[dataframe['atributo'] == atributo].iloc[:, 0])
            resultados.append((modelo, 't', atributo, t_resultado.statistic, t_resultado.pvalue))
        elif atributo.startswith('var_'):
            f_resultado = stats.f_oneway(globals_copy['atributos_well'][globals_copy['atributos_well']['atributo'] == atributo].iloc[:, 0], dataframe[dataframe['atributo'] == atributo].iloc[:, 0])
            resultados.append((modelo, 'f', atributo, f_resultado.statistic, f_resultado.pvalue))

# 'resultados' é uma lista de tuplas, onde cada tupla contém o nome do modelo, o tipo de teste ('t' ou 'f'), o nome do atributo, a estatística do teste e o valor-p
resultados_testes_estatisticos = pd.DataFrame(resultados, columns=['modelo', 'teste', 'atributo', 'estatistica', 'valor_p'])




# testes t


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

########################################################################
########################################################################
########################################################################
## plotagem dos perfis

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



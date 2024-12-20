# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 17:04:58 2024

@author: 02901717926
"""
import time
import pandas as pd
from scipy.stats import ttest_ind, f_oneway
import numpy as np
import matplotlib.pyplot as plt



# Marca o tempo de início
inicio = time.time()

# Carregar dados do arquivo de entrada em txt
# df = pd.read_csv('https://raw.githubusercontent.com/danielbettu/danielbettu/main/eaton_Gov_Gp_Gc_Gf.txt', sep= "\t", header=None)
df = pd.read_csv('C:/Python/PROOF_pocos_validacao.csv', sep= ",", header= 0)
# df = pd.read_csv('/home/bettu/Documents/Python/PROOF/PROOF_pocos_validacao.csv', sep= ",", header= 0)

#######################################################################
#######################################################################
#######################################################################
# Classe Litológica Textural Média


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

atrib_litho_dom = df.groupby('layer')[cols].agg(most_frequent_and_max).reset_index() # litologia dominante em cada layer
atrib_litho_dom.columns = atrib_litho_dom.columns.str.replace('_text', '')
    
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
atrib_terr_prop = df.groupby('layer')[cols].apply(calc_prop).reset_index()
atrib_terr_prop.columns = atrib_terr_prop.columns.str.replace('_comp', '')
    
########################################################################
########################################################################
########################################################################
## faixa batimétrica média

# Lista de colunas para calcular a batimetria média dos layers
cols = ['well_bat', 'M1_bat', 'M2_bat', 'M3_bat', 'M4_bat']

# Calcula a média para cada layer e cria um novo DataFrame
atrib_bat_mean = df.groupby('layer')[cols].mean()
# Remove o sufixo '_bat' dos nomes das colunas
atrib_bat_mean.columns = atrib_bat_mean.columns.str.replace('_bat', '')
# Adiciona uma nova coluna contendo o número dos layers
atrib_bat_mean.insert(0, 'layer_number', atrib_bat_mean.index)
atrib_bat_mean.columns = atrib_bat_mean.columns.str.replace('_number', '')

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

'''
s.shift(): Este método desloca os índices de uma Série pandas 's' por um número especificado de períodos, que é 1 por padrão. 
Isso significa que cada elemento em 's' é movido para baixo por uma posição, e o primeiro elemento é substituído por NaN.

s != s.shift(): Esta linha de código compara cada elemento em 's' com o elemento abaixo dele (devido ao deslocamento). 
Se os dois elementos forem diferentes, o resultado é True; se forem iguais, o resultado é False. 
Isso retorna uma nova Série de valores booleanos do mesmo tamanho que 's'.

(s != s.shift()).cumsum(): O método cumsum() é aplicado à Série booleana, que retorna a soma cumulativa dos valores. 
Em outras palavras, ele adiciona os valores True (considerados como 1) à medida que avança pela Série.

(s != s.shift()).cumsum().value_counts(): Por fim, o método value_counts() conta a ocorrência de cada valor único na Série cumsum. 
Isso retorna uma nova Série onde o índice é o valor único da Série original e o valor é a contagem de ocorrências desse valor.
Em resumo, este código está contando a frequência de sequências consecutivas de valores idênticos na Série 's'.
'''
mean_values = {}

for key, df_mean_thick in dfs.items():  # Altera df para df_mean_thick
    counts = {}
    for col in cols:
        sequences = count_sequences(df_mean_thick[col])  # Altera df para df_mean_thick
        counts[col] = sequences.mean()
    mean_values[key] = counts

# Convertendo o dicionário em um DataFrame
mean_thick = pd.DataFrame(mean_values).T
atrib_sqrt_mean_thick = mean_thick.copy()
atrib_sqrt_mean_thick.columns = atrib_sqrt_mean_thick.columns.str.replace('_comp', '')
atrib_sqrt_mean_thick = atrib_sqrt_mean_thick.applymap(np.sqrt) # Atualiza todas as colunas do dataframe atrib_sqrt_mean_thick para que contenham o valor da raiz quadrada do valor original
# Adiciona uma nova coluna contendo o número dos layers
atrib_sqrt_mean_thick.insert(0, 'layer_number', atrib_sqrt_mean_thick.index)
atrib_sqrt_mean_thick.columns = atrib_sqrt_mean_thick.columns.str.replace('_number', '')

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
'''
# Inicializa o dicionário final
ang_coef_dict = {}

# cálculo do coeficiente angular - Loop através de cada coluna/modelo
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
    
# Criando o DataFrame transposto com índice
atrib_text_trend = pd.DataFrame.from_dict(ang_coef_dict, orient='index')
atrib_text_trend = atrib_text_trend.rename(columns={atrib_text_trend.columns[0]: 'text_trend'})
atrib_text_trend =atrib_text_trend.T
atrib_text_trend.columns = atrib_text_trend.columns.str.replace('_text', '')
'''
# Criando uma coluna 'group' com base na numeração de 'layer'
df_ang_coef['group'] = df_ang_coef['layer'].apply(lambda x: 1 if x in [1, 2, 3, 4, 5] else 2)

# Inicializa o dicionário final para cada grupo
ang_coef_dict_group1 = {}
ang_coef_dict_group2 = {}

# Loop através de cada coluna/modelo
for col in cols:
    # Filtra os dados para cada grupo
    group1_data = df_ang_coef[df_ang_coef['group'] == 1][col]
    group2_data = df_ang_coef[df_ang_coef['group'] == 2][col]
    
    # Cria DataFrames com os valores para cada grupo
    df_b_coeff_group1 = pd.DataFrame({'y': group1_data, 'x': range(len(group1_data))})
    df_b_coeff_group2 = pd.DataFrame({'y': group2_data, 'x': range(len(group2_data))})
    
    # Calcula a covariância entre x e y para cada grupo
    covariance_group1 = df_b_coeff_group1['x'].cov(df_b_coeff_group1['y'])
    covariance_group2 = df_b_coeff_group2['x'].cov(df_b_coeff_group2['y'])
    
    # Calcula a variância de x para cada grupo
    variance_group1 = df_b_coeff_group1['x'].var()
    variance_group2 = df_b_coeff_group2['x'].var()
    
    # Coeficientes angulares para cada grupo
    angular_coeff_group1 = covariance_group1 / variance_group1
    angular_coeff_group2 = covariance_group2 / variance_group2
    
    # Adiciona ao dicionário final para cada grupo
    ang_coef_dict_group1[col] = angular_coeff_group1
    ang_coef_dict_group2[col] = angular_coeff_group2

# Criando os DataFrames transpostos com índice para cada grupo
atributo_text_trend_UP = pd.DataFrame.from_dict(ang_coef_dict_group1, orient='index')
atributo_text_trend_LW = pd.DataFrame.from_dict(ang_coef_dict_group2, orient='index')

# Renomeando as colunas
atributo_text_trend_UP = atributo_text_trend_UP.rename(columns={atributo_text_trend_UP.columns[0]: 'text_trend_group1'})
atributo_text_trend_LW = atributo_text_trend_LW.rename(columns={atributo_text_trend_LW.columns[0]: 'text_trend_group2'})

# Concatenando os resultados finais
atrib_text_trend = pd.concat([atributo_text_trend_UP, atributo_text_trend_LW], axis=1)
atrib_text_trend =  atrib_text_trend.T
atrib_text_trend.columns = atrib_text_trend.columns.str.replace('_text', '')


# Exibindo o resultado
atrib_text_trend

# # ########################################################################
# # ########################################################################
# # ########################################################################
# # ## Variabilidade litológica - frequência de transição de litologia

# Define as colunas que representam os modelos
cols = ['well_comp', 'M1_comp', 'M2_comp', 'M3_comp', 'M4_comp']

# Cria uma cópia do DataFrame original para calcular as mudanças
df_litho_change = df.copy()

# cálculo da variabilidade média 

# Lista para armazenar os DataFrames temporários
temp_dfs = []

# Loop sobre as colunas
for col in cols:
    # Agrupa os resultados pela coluna 'layer'
    grouped_by_layer = df_litho_change.groupby('layer')[col]

    # Calcula a diferença entre cada valor e o valor anterior dentro de cada grupo
    diff_by_layer = grouped_by_layer.diff()

    # Desconsidera a primeira linha de cada camada
    diff_by_layer[df_litho_change['layer'] != df_litho_change['layer'].shift()] = 0

    # Conta o número de mudanças dentro de cada camada
    change_count_by_layer = diff_by_layer.ne(0).groupby(df_litho_change['layer']).sum()

    # Cria um DataFrame temporário
    temp_df = pd.DataFrame({
        f'{col}_change_count_by_layer': change_count_by_layer
    })

    # Adiciona o DataFrame temporário à lista
    temp_dfs.append(temp_df)

# Combina os DataFrames temporários em um único DataFrame
atrib_variab_litho = pd.concat(temp_dfs, axis=1)

# Adiciona a coluna 'layer' na primeira posição do DataFrame final
atrib_variab_litho.insert(0, 'layer', change_count_by_layer.index)
atrib_variab_litho.columns = atrib_variab_litho.columns.str.replace('_comp_change_count_by_layer', '')

########################################################################
########################################################################
########################################################################
## agrupando os dataframes em um dicionário de atributos por layer

# Obtém uma cópia de todas as variáveis no ambiente de trabalho
all_variables = dict(globals())

# Dicionário para armazenar os DataFrames que começam com "atrib_"
atributos_dict = {}

# Lista para armazenar temporariamente os nomes das variáveis que atendem à condição
atrib_names = []

# Coleta os nomes das variáveis que atendem à condição
for var_name in all_variables:
    if var_name.startswith("atrib_") and isinstance(all_variables[var_name], pd.DataFrame):
        atrib_names.append(var_name)

# Agora, você pode iterar sobre a lista de nomes para criar o dicionário
for df_name in atrib_names:
    # Adiciona o DataFrame ao dicionário
    atributos_dict[df_name] = all_variables[df_name]

########################################################################
########################################################################
########################################################################
## testes estatísticos

# Dicionário para armazenar os resultados dos testes
statistical_tests_results = {}

# Loop sobre cada DataFrame no dicionário
for df_name, df_atributos in atributos_dict.items():
    # Lista para armazenar os resultados dos testes para cada par de modelo
    resultados_modelos = []

    # Loop sobre as colunas do DataFrame (exceto 'well')
    for col in df_atributos.columns[1:]:
        # Obtém os dados do modelo de referência e do modelo atual
        dados_referencia = df_atributos['well']
        dados_modelo = df_atributos[col]

        # Teste t (t-test)
        t_statistic, t_p_value = ttest_ind(dados_referencia, dados_modelo)

        # Teste F (ANOVA)
        f_statistic, f_p_value = f_oneway(dados_referencia, dados_modelo)

        # Armazena os resultados
        resultados_modelos.append({
            'modelo_referencia': 'well',
            'modelo_atual': col,
            't_statistic': t_statistic,
            't_p_value': t_p_value,
            'f_statistic': f_statistic,
            'f_p_value': f_p_value
        })

    # Adiciona os resultados ao dicionário geral
    statistical_tests_results[df_name] = resultados_modelos
   
########################################################################
########################################################################
########################################################################
## Organização dos resultados

resultado_sintetizado = {}

for atributo, modelos in statistical_tests_results.items():
    resultado_sintetizado[atributo] = {}
    for modelo in modelos:
        modelo_atual = modelo['modelo_atual']
        if modelo_atual not in resultado_sintetizado[atributo]:
            resultado_sintetizado[atributo][modelo_atual] = {
                't_statistic': [],
                't_p_value': [],
                'f_statistic': [],
                'f_p_value': [],
            }
        resultado_sintetizado[atributo][modelo_atual]['t_statistic'].append(modelo['t_statistic'])
        resultado_sintetizado[atributo][modelo_atual]['t_p_value'].append(modelo['t_p_value'])
        resultado_sintetizado[atributo][modelo_atual]['f_statistic'].append(modelo['f_statistic'])
        resultado_sintetizado[atributo][modelo_atual]['f_p_value'].append(modelo['f_p_value'])

# Criando o DataFrame
rows = []

for atributo, modelos in resultado_sintetizado.items():
    for modelo_atual, estatisticas in modelos.items():
        row = {'Atributo': atributo, 'Modelo Atual': modelo_atual}
        row.update({f'Média {key}': sum(values) / len(values) for key, values in estatisticas.items()})
        rows.append(row)

statistical_results_final = pd.DataFrame(rows)
statistical_results_final.columns = statistical_results_final.columns.str.replace('Média ', '')

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



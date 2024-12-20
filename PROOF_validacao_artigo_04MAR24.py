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
import math

# Marca o tempo de início
inicio = time.time()

# Carregar dados do arquivo de entrada em txt
# df = pd.read_csv('https://raw.githubusercontent.com/danielbettu/danielbettu/main/eaton_Gov_Gp_Gc_Gf.txt', sep= "\t", header=None)
df = pd.read_csv('C:/Python/PROOF_pocos_validacao.csv', sep= ";", header= 0)
# df = pd.read_csv('/home/bettu/Documents/Python/PROOF/PROOF_pocos_validacao.csv', sep= ",", header= 0)

list_text = [col for col in df.columns if "_text" in col]
list_comp = [col for col in df.columns if "_comp" in col]
list_bat = [col for col in df.columns if "_bat" in col]

#######################################################################
#######################################################################
#######################################################################
# Classe Litológica Textural Média

# cálculo de espessura dos layers 
df_litho = df.copy()
# grouped = df_litho.groupby('layer')
thick_sum = df_litho['espessura'].sum() # Calculando o somatório da coluna 'espessura' 

cols = list_text.copy()

atributo = df_litho[cols]
espessura = df_litho['espessura']
atributo = pd.concat([df_litho[cols], espessura], axis=1)
peso_col = atributo['espessura']

def calcular_media_ponderada(atributo, cols, peso_col):
    medias_ponderadas = {}
    for col in cols:
        medias_ponderadas[col] = np.average(atributo[col], weights=peso_col)
    return medias_ponderadas

# Aplicação da função calcular_media_ponderada
pre_mean_litho_dom = calcular_media_ponderada(atributo, cols, peso_col)

def calcular_variancia_ponderada(atributo, cols, peso_col, medias):
    variancias_ponderadas = {}
    for col in cols:
        diff = atributo[col] - medias[col]
        variancias_ponderadas[col] = np.average(diff**2, weights=atributo[peso_col])
    return variancias_ponderadas

# Aplicação da função calcular_variancia_ponderada
pre_var_litho_dom = calcular_variancia_ponderada(atributo, cols, 'espessura', pre_mean_litho_dom)

#renomeando as linhas do dict
mean_litho_dom = {}
for key, value in pre_mean_litho_dom.items():
    new_key = key.replace("_text", "")  # Remove '_text' da chave
    mean_litho_dom[new_key] = value

# var_litho_dom = var_litho_dom.copy()
var_litho_dom = {}
for key, value in pre_var_litho_dom.items():
    new_key = key.replace("_text", "")  # Remove '_text' da chave
    var_litho_dom[new_key] = value

########################################################################
########################################################################
########################################################################
## Proporção de Terrígenos média do poço 

# Função para calcular a proporção de valores ímpares (terrígenos = ímpar; carbonatos = par)

df_terr_prop = df.copy()

cols = list_comp.copy()

# Calcular a razão para cada coluna
razoes = {}
for col in cols:
    valores = df_terr_prop[col]
    impares = valores[valores % 2 != 0]
    razao = len(impares) / len(valores)
    razoes[col] = razao

# Calcular a média ponderada
pre_mean_terr_prop = calcular_media_ponderada(razoes, cols, len(valores))

# calculando a variância  
# Novo dicionário
pre_var_terr_prop = {}
# Iterar sobre o dicionário original
for key, value in pre_mean_terr_prop.items():
    # Calcular o novo valor
    new_value = value * (1 - value)
    
    # Verificar se new_value é menor que 0.2
    if new_value < 0.2:
        new_value = 0.2
    
    # Adicionar ao novo dicionário
    pre_var_terr_prop[key] = new_value
    
#renomeando as linhas do dict
mean_terr_prop = {}
for key, value in pre_mean_terr_prop.items():
    new_key = key.replace("_comp", "")  # Remove '_comp' da chave
    mean_terr_prop[new_key] = value

# var_litho_dom = var_litho_dom.copy()
var_terr_prop = {}
for key, value in pre_var_terr_prop.items():
    new_key = key.replace("_comp", "")  # Remove '_comp' da chave
    var_terr_prop[new_key] = value

########################################################################
########################################################################
########################################################################
## faixa batimétrica média do poço

cols = list_bat.copy()

mean_bat = df[cols].mean() # Calcula a média para o poço e cria um novo DataFrame
var_bat = df[cols].var() # calcula a variância
mean_bat.index = mean_bat.index.str.replace('_bat', '')
var_bat.index = var_bat.index.str.replace('_bat', '')

########################################################################
########################################################################
########################################################################
## espessura média das camadas no poço

cols = list_comp.copy()

# Função para contar sequências consecutivas
def count_sequences(s):
    return (s != s.shift()).cumsum().value_counts()

# Dicionário para armazenar os valores médios
dict_mean_values = {}
temp_dfs = {}  # Dicionário para armazenar os DataFrames temporários

for col in cols:
    # Calcula as sequências para cada coluna
    sequences = count_sequences(df[col])
    
    # Calcula a espessura acumulada para cada sequência
    accumulated_thickness = sequences * df['espessura']
    
    # Calcula a raiz quadrada da espessura acumulada
    sqrt_accumulated_thickness = np.sqrt(accumulated_thickness)
    
    # Calcula a média da raiz quadrada da espessura
    dict_mean_values[col] = sqrt_accumulated_thickness.mean()
    
    # Cria um DataFrame temporário para cada coluna
    temp_dfs[col] = pd.DataFrame({
        'Sequences': sequences,
        'Accumulated_Thickness': accumulated_thickness,
        'Sqrt_Thickness': sqrt_accumulated_thickness
    })

# Cria uma série com os valores médios
mean_sqrt_thickness = pd.Series(dict_mean_values)
mean_sqrt_thickness.index = mean_sqrt_thickness.index.str.replace('_comp', '')

# Combine os DataFrames temporários em um único DataFrame
combined_temp_df = pd.concat(temp_dfs.values(), axis=1)

#######################################################################
#######################################################################
#######################################################################
# tendência de variação granulométrica vertical

# Classes texturais 1, 3 e 5

df_ang_coef = df.copy()

# # Criando uma coluna 'group' com base na numeração de 'layer'
# valores_unicos_layer = df_ang_coef['layer'].unique()

# Calcule a mediana dos valores únicos da coluna 'layer'
profundidade_mediana = thick_sum/2

# Ajuste a coluna 'group' com base na mediana
# df_ang_coef['group'] = df_ang_coef['layer'].apply(lambda x: 1 if x <= profundidade_mediana else 2)
df_ang_coef['group'] = df_ang_coef.index.to_series().apply(lambda x: 1 if x <= profundidade_mediana else 2)


# Inicializa o dicionário final para cada grupo
ang_coef_dict_group1 = {}
ang_coef_dict_group2 = {}

cols = list_text.copy()

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
atrib_text_trend.insert(0, 'layer', range(1, len(atrib_text_trend) + 1)) # Adicionar a primeira coluna 'layer' e definir a primeira linha como 1 e a segunda como 5
atrib_text_trend['layer'].iloc[1] = float(profundidade_mediana) # incluido o valor profundidade_mediana (primeira linha da seção LW) nessa posição para diferenciar a camada UP de LW no resultado final

# Obtenha os nomes das linhas como seus intervalos
intervalos = atrib_text_trend.index.tolist()
modelos = list(atrib_text_trend.columns)

# Criação dos novos dataframes
dataframes = {}
for intervalo in intervalos:
    df_temp = pd.DataFrame(atrib_text_trend.loc[intervalo, modelos])
    df_temp.columns = ['Value']
    df_temp.index.name = 'Model'
    df_temp.reset_index(inplace=True)
    dataframes[f'pre_{intervalo}'] = df_temp

pre_mean_trend_text_UP = dataframes['pre_text_trend_group1']
pre_mean_trend_text_UP.set_index('Model', inplace=True)
first_index = pre_mean_trend_text_UP.index[0] # Obtenha o nome do índice da primeira linha
pre_mean_trend_text_UP.drop(first_index, inplace=True) # Remova a primeira linha
mean_trend_text_UP = pre_mean_trend_text_UP['Value']

pre_mean_trend_text_LW = dataframes['pre_text_trend_group2']
pre_mean_trend_text_LW.set_index('Model', inplace=True)
first_index = pre_mean_trend_text_LW.index[0] # Obtenha o nome do índice da primeira linha
pre_mean_trend_text_LW.drop(first_index, inplace=True) # Remova a primeira linha
mean_trend_text_LW = pre_mean_trend_text_LW['Value']

# # ########################################################################
# # ########################################################################
# # ########################################################################
# # ## Variabilidade litológica - frequência de transição de litologia

# Create a copy of df_litho_change
df_litho_change = df.copy()

# List to store temporary DataFrames
temp_dfs = []

# Columns of interest
cols = list_comp.copy()

# Loop over the columns
for col in cols:
    # Calculate the difference between each value and the previous value within each group
    diff_by_layer = df_litho_change[col].diff()

    # Disregard the first row of each layer
    diff_by_layer[df_litho_change['layer'] != df_litho_change['layer'].shift()] = 0

    # Count the number of changes within each layer
    change_count_by_layer = diff_by_layer.ne(0).sum()

    # Create a temporary DataFrame with an index
    temp_df = pd.DataFrame({
        f'{col}_change_count_by_layer': change_count_by_layer
    }, index=df_litho_change.index)

    # Add the temporary DataFrame to the list
    temp_dfs.append(temp_df)

# Combine the temporary DataFrames into a single DataFrame
atrib_variab_litho = pd.concat(temp_dfs, axis=1)
atrib_variab_litho.columns = atrib_variab_litho.columns.str.replace('_comp_change_count_by_layer', '')
atrib_variab_litho = atrib_variab_litho.T
numero_transicoes = atrib_variab_litho[0].squeeze()
numero_transicoes -= 1
# Calcula o comprimento da série (número de elementos)
length_numero_transicoes = len(df)
# Calcula a nova série 'mean_litho_trans'
mean_litho_trans = numero_transicoes / (length_numero_transicoes - 1)
var_litho_trans = mean_litho_trans * (1 - mean_litho_trans)

########################################################################
########################################################################
########################################################################
## agrupando as séries de médias e atributos 

# Filtrar as variáveis globais que começam com 'mean_' ou 'var_'
selected_variables = {}

# Criar cópia do dicionário antes de iterar
globals_copy = dict(globals())

for k, v in globals_copy.items():
    if isinstance(v, (pd.Series, dict)) and (k.startswith('mean_') or k.startswith('var_')):
        selected_variables[k] = v

# Criar DataFrame a partir do dicionário de variáveis
res_mean_var_atributos = pd.DataFrame(selected_variables)

########################################################################
########################################################################
########################################################################
## testes estatísticos

# Definindo o índice
res_mean_var_atributos = res_mean_var_atributos.T

# Lista de atributos
atributos = res_mean_var_atributos.index.tolist()
modelos = res_mean_var_atributos.columns.tolist()

# Inicializar valor_p do teste F fora do loop for
p_valor_teste_F = None

# Lista para armazenar DataFrames temporários
resultados_temporarios = []

# Loop for para testes t e F
for modelo in modelos:
    for atributo in atributos:
        if atributo.startswith('mean_'):
            temp = atributo.replace('mean_', 'var_')
            media_amostra1 = res_mean_var_atributos.loc[atributo, 'well']
            if temp in res_mean_var_atributos.index:
                variancia_amostra1 = res_mean_var_atributos.loc[temp, 'well']
                if variancia_amostra1 < 0.2:
                    variancia_amostra1 = 0.2
            else:
                variancia_amostra1 = 0.2
            tamanho_amostra1 = len(df)
            
            media_amostra2 = res_mean_var_atributos.loc[atributo, modelo]
            if temp in res_mean_var_atributos.index:
                variancia_amostra2 = res_mean_var_atributos.loc[temp, modelo]
                if variancia_amostra2 < 0.2:
                    variancia_amostra2 = 0.2
            tamanho_amostra2 = len(df)

            # Teste t alternativo            
            # estatistica_t = abs(media_amostra1 - media_amostra2) / math.sqrt((variancia_amostra1 / tamanho_amostra1) + (variancia_amostra2 / tamanho_amostra2))
            # graus_de_liberdade = tamanho_amostra1 + tamanho_amostra2 - 2
            # p_valor_teste_t = stats.t.sf(abs(estatistica_t), graus_de_liberdade) * 2
            
            #### Teste t relatório
            estatistica_t = abs(media_amostra1 - media_amostra2) / math.sqrt((variancia_amostra1 + variancia_amostra2)/(2*(tamanho_amostra1-1)))
            graus_de_liberdade = tamanho_amostra1 + tamanho_amostra2 - 2
            p_valor_teste_t = stats.t.sf(abs(estatistica_t), graus_de_liberdade) * 2
            
            # Criar DataFrame temporário
            temp_df = pd.DataFrame({'Modelo': [modelo], 'Atributo': [atributo], 'Estatistica_T': [estatistica_t], 'p_valor_t': [p_valor_teste_t]})

        elif atributo.startswith('var_'):
            # Calcule a variância da amostra para o atributo 'well'
            variancia_amostra1 = res_mean_var_atributos.loc[atributo, 'well']
            if variancia_amostra1 < 0.2:
                variancia_amostra1 = 0.2
            
            # Calcule a variância da amostra para o atributo 'modelo'
            # Se o valor for menor que 0.2, defina como 0.2
            variancia_amostra2 = res_mean_var_atributos.loc[atributo, modelo] if (res_mean_var_atributos.loc[atributo, modelo] >= 0.2) else 0.2
            if variancia_amostra2 < 0.2:
                variancia_amostra2 = 0.2
            
            # Calcule a estatística F
            if variancia_amostra1 >= variancia_amostra2:
                estatistica_F = variancia_amostra1 / variancia_amostra2
            else:
                estatistica_F = variancia_amostra2 / variancia_amostra1
                
            if estatistica_F >= 3.2:
                estatistica_F = 3.2
            if estatistica_F <= 1.2:
                estatistica_F = 1.2
            
            # Graus de liberdade
            graus_liberdade1 = len(df) - 1
            graus_liberdade2 = len(df) - 1
            
            # Calcule o valor-p do teste F
            p_valor_teste_F = stats.f.cdf(estatistica_F, graus_liberdade1, graus_liberdade2)
            
            # Crie um DataFrame temporário
            temp_df = pd.DataFrame({
                'Modelo': [modelo],
                'Atributo': [atributo],
                'Estatistica_F': [estatistica_F],
                'p_valor_F': [p_valor_teste_F]
            })
       
        # Adicionar DataFrame temporário à lista
        resultados_temporarios.append(temp_df)

# Concatenar todos os DataFrames temporários em um único DataFrame
resultados_estatisticos_comparacao_well = pd.concat(resultados_temporarios, ignore_index=True)

########################################################################
########################################################################
########################################################################
## Combinação das probabilidades

# Dicionário para armazenar os dataframes
resultados_estatisticos_consolidados = {}

# Itera pelos modelos
for modelo in modelos:
    # Filtra os dados para o modelo atual
    dados_modelo = resultados_estatisticos_comparacao_well[resultados_estatisticos_comparacao_well['Modelo'] == modelo]
    
    # Cria o dataframe com os atributos e p_valor
    df_modelo = pd.DataFrame({
        'Atributo': atributos,
        'p_valor': dados_modelo.apply(lambda row: row['p_valor_t'] if row['Atributo'].startswith('mean_') else row['p_valor_F'], axis=1)
    })
    
    # Adiciona o dataframe ao dicionário
    resultados_estatisticos_consolidados[modelo] = df_modelo
    
# Regra para ajustar os valores de p_valor
def ajustar_p_valor(valor):
    if valor < 0.0052:
        return 0.0052
    elif valor > 0.92151:
        return 0.92151
    else:
        return valor

# Itera pelos dataframes no dicionário
for modelo, df_temp in resultados_estatisticos_consolidados.items():
    df_temp['p_valor'] = df_temp['p_valor'].apply(ajustar_p_valor)    

########################################################################
########################################################################
########################################################################
## Combinação das probabilidades

# Inicialize o DataFrame de resultados
resultados = pd.DataFrame(columns=['modelo_atual', 'prob(U|E)', 'prob(nU|E)'])

# Itera pelos modelos
for modelo, sub_df in resultados_estatisticos_consolidados.items():
    # Inicializar variáveis para armazenar o resultado do produtório
    prob_inicial = None

    # Iterar sobre cada linha do sub_df
    for index, row in sub_df.iterrows():
        # Calcular prob_adicional
        prob_adicional = row['p_valor']

        # Se prob_inicial ainda não foi definida, defina-a como a primeira ocorrência de p_valor
        if prob_inicial is None:
            prob_inicial = prob_adicional
            continue

        # Calcular 'prob(U|E)' e 'prob(nU|E)'
        prob_U_E = prob_inicial * prob_adicional
        prob_nU_E = (1 - prob_inicial) * (1 - prob_adicional)

        # Calcular prob_somada
        prob_somada = prob_U_E + prob_nU_E

        # Normalizar prob(U|E) e prob(nU|E)
        prob_U_E /= prob_somada
        prob_nU_E /= prob_somada

        # Usar a nova prob(U|E) normalizada como nova prob_inicial
        prob_inicial = prob_U_E

    # Criar um novo DataFrame com os resultados do modelo atual
    df_modelo = pd.DataFrame({'modelo_atual': [modelo], 'prob(U|E)': [prob_U_E], 'prob(nU|E)': [prob_nU_E]})

    # Concatenar o novo DataFrame com o DataFrame de resultados
    resultados = pd.concat([resultados, df_modelo], ignore_index=True)

# Resultados intermediários
resultados_intermediarios = resultados.copy()

######################################################
# Subtrair um valor específico de 'resultado' e manter a primeira coluna de strings intacta
valor_a_subtrair = 1
resultados_intermediarios['prob(U|E)'] = np.where(resultados_intermediarios['prob(U|E)'].apply(type) == str, resultados_intermediarios['prob(U|E)'], valor_a_subtrair - resultados_intermediarios['prob(U|E)'] )

# Atribuir o resultado a PROOF
PROOF = resultados_intermediarios
PROOF = PROOF.rename(columns={'prob(U|E)': 'PROOF'})
PROOF = PROOF.drop('prob(nU|E)', axis=1)

########################################################################
########################################################################
########################################################################
## plotagem dos perfis

cols = list_comp.copy()

# Mapeamento de cores
cores = {1: 'green', 2: 'lightblue', 3: 'yellow', 4: 'blue', 5: 'orange', 6: 'darkblue'}

# Criação da figura e dos eixos
fig, axs = plt.subplots(1, len(cols), figsize=(len(cols)*3, 20))

# Loop sobre cada coluna
for i, col in enumerate(cols):
    # Seleciona os dados
    x = df[col]
    y = df['profundidade']
    
    # Obtém o valor correspondente de PROOF para o modelo atual
    modelo_atual = col.replace('_comp', '')  # Remove '_comp' do nome da coluna
    proof_value = PROOF.loc[PROOF['modelo_atual'] == modelo_atual, 'PROOF'].values[0]
    
    # Cria o gráfico de barras com as cores correspondentes
    axs[i].barh(y, x, color=[cores[val] for val in x])
    
    # Inverte o eixo y para que a profundidade aumente para baixo
    axs[i].invert_yaxis()
    
    # Define o título do subplot com o valor de PROOF (3 casas decimais)
    axs[i].set_title(f"{col} (PROOF: {proof_value:.3f})")
    
    # Define os intervalos dos eixos
    axs[i].set_yticks(np.arange(0, max(y) + 1, 1))
    axs[i].set_xticks(np.arange(0, max(x) + 1, 1))
# Mostra a figura
plt.tight_layout()
plt.show()

# Marca o tempo de término
fim = time.time()
# Calcula a diferença
tempo_decorrido = fim - inicio
print(f"O tempo decorrido foi de {tempo_decorrido} segundos.")
print(PROOF)

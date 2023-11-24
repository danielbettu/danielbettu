# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 18:23:07 2023

@author: 02901717926
"""

import pandas as pd
import numpy as np

#códigos para carregamento de dados diversos em dataframe

# Carreguar dados do arquivo de entrada em txt
dados = np.loadtxt('C:/Python/p-lutite.txt')

#OU

# Carreguar dados do arquivo de entrada em txt
df = pd.read_csv('C:/Python/eaton_Gov_Gp_Gc_Gf.txt', sep= "\t", header=None) #melhor porque mantém os dados no formato original

# Caso queira carregar apenas uma das colunas ou linhas do dado acima
col1 = dados[:, 0]
col2 = dados[:, 1]
lin1 = dados[1, :]

# Carreguar dados do arquivo de entrada em xlsx - mensagem de que irá ignorar cabeçalho (se houver)
df = pd.read_excel('c:\Python\p_lutite.xlsx', engine='openpyxl')

# Carreguar dados do arquivo de entrada em xls
df = pd.read_excel('C:/Python/fft_teste.xls', engine='xlrd')


#######################################################################

# Carregar imagem .tif
import cv2
# Carregue a imagem
img = cv2.imread('C:/Python/srtm_30m.tif')

# Converta a imagem em um array numpy
img_array = np.array(img)


# Carregue a imagem
img = cv2.imread('C:/Python/srtm_30m.tif')

# Exiba a imagem
cv2.imshow('Imagem', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


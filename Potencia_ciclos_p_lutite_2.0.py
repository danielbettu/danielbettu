#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 22:23:20 2023

@author: bettu
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Carregar os dados do arquivo .xlsx usando o pacote pandas com o motor 'openpyxl'
df = pd.read_excel('/home/bettu/Documents/Python/teste_fft.xlsx', engine='openpyxl')

# Definir as unidades das colunas
age_unit = 'Years'
prop_unit = 'Proportion'

# Renomear as colunas, se necessário
df = df.rename(columns={'Idades': 'AGE', 'p_lutite': 'PROP'})

# Plotar o gráfico da série temporal
plt.plot(df['AGE'], df['PROP'])
plt.xlabel('Age ({})'.format(age_unit))
plt.ylabel('Proportion ({})'.format(prop_unit))
plt.title('Variation of Proportion over Time')
plt.grid(True)
plt.show()

# Realizar a decomposição da série temporal usando FFT
prop_values = df['PROP'].values
n = len(prop_values)
timestep = df['AGE'][1] - df['AGE'][0]  # Assumindo que os dados têm espaçamento de tempo regular

# Aplicar FFT nos dados
fft_values = fft(prop_values)
freqs = fftfreq(n, d=timestep)

# Calcular a densidade espectral de potência (PSD)
power = np.abs(fft_values) ** 2

# Identificar as frequências de variação
positive_freqs = freqs[:n//2]
positive_power = power[:n//2]

# Plotar a potência versus a duração dos ciclos até 150000 anos
cycle_duration = 1 / positive_freqs  # Duração dos ciclos (reciprocals das frequências)
plt.figure(figsize=(20, 11))
plt.plot(cycle_duration, positive_power)
plt.xlim(0,150000)
plt.ylim(0, 5)
plt.xlabel('Cycle Duration ({})'.format(age_unit))
plt.ylabel('Power')
plt.title('Power Spectrum (Cycles up to 150,000 Years)')
plt.grid(True)
plt.show()

# Imprimir os ciclos identificados e potências
print("Cycles and Power:")
for cycle, power in zip(cycle_duration, positive_power):
    print("Cycle Duration: {:.2f} {} | Power: {:.2f}".format(cycle, age_unit, power))

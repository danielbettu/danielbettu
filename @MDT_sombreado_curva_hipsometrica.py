# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:25:57 2023

@author: 02901717926
"""

import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib import cm
import numpy as np

# Carregue a imagem
with rasterio.open('C:/Python/srtm_30m.tif') as src:
    # Leia a primeira banda do conjunto de dados raster (assumindo que seu arquivo .tif tem apenas uma banda)
    elevation = src.read(1)

# Crie um objeto de fonte de luz
ls = LightSource(azdeg=90, altdeg=60)

# Crie um relevo sombreado
shaded = ls.shade(elevation, vert_exag=1, cmap=cm.terrain, blend_mode='overlay')

# Exiba o relevo sombreado
plt.imshow(shaded, cmap='gray')
plt.show()


#########################

# Calcule a curva hipsométrica
unique, counts = np.unique(elevation, return_counts=True)

# Inverta a ordem dos dados
unique = unique[::-1]
counts = counts[::-1]

# Calcule a soma cumulativa
cumulative_counts = np.cumsum(counts)

plt.figure()
plt.plot(cumulative_counts, unique)
plt.ylabel('Elevação (m)')
plt.xlabel('Área acumulada (m2)')
plt.title('Curva Hipsométrica')
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:21:00 2024

@author: bettu
"""

import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
import pandas as pd

def extrair_emails(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            conteudo = response.text
            soup = BeautifulSoup(conteudo, 'html.parser')
            texto = soup.get_text()

            # Expressão regular aprimorada para capturar mais emails
            emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', texto)
            emails = list(set(emails))
            
            return emails
        else:
            print(f"Falha ao acessar a URL: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Erro na requisição: {e}")
        return []

url = 'https://termocronologiaunesp.wixsite.com/termocronologiaunesp/members'
emails = extrair_emails(url)
df = pd.DataFrame(emails, columns=['Emails'])
print(df)
df.to_clipboard(index=False) 
print("DataFrame copiado para a área de transferência!")

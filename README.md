# FronteiraEficiente
Fronteira Eficiente de Markowitz em Python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime
acoes = ['VVAR3.SA', 'COGN3.SA', 'BBDC4.SA']
dados = web.DataReader(acoes, 'yahoo', '2015-1-1')['Adj Close']

#Calcular Retorno
retorno_diario = dados.pct_change()
retorno_anual = retorno_diario.mean() * 250

#Calcular Covariancia
cov_diaria = retorno_diario.cov()
cov_anual = cov_diaria * 250

#Variaveis
retorno_carteira = []
peso_acoes = []
volatilidade_carteira = []
sharpe_ratio = []

#Simular n portfolios
numero_acoes = len(acoes)
numero_carteiras = 100000

np.random.seed(101)

for cada_carteira in range(numero_carteiras):
    peso = np.random.random(numero_acoes)
    peso /= np.sum(peso)
    retorno = np.dot(peso, retorno_anual)
    volatilidade = np.sqrt(np.dot(peso.T, np.dot(cov_anual, peso)))
    sharpe = retorno / volatilidade
    sharpe_ratio.append(sharpe)
    retorno_carteira.append(retorno)
    volatilidade_carteira.append(volatilidade)
    peso_acoes.append(peso)

#dicionario com dados
carteira = {'Retorno': retorno_carteira,
             'Volatilidade': volatilidade_carteira,
             'Sharpe Ratio': sharpe_ratio}

for contar,acao in enumerate(acoes):
    carteira[acao+' Peso'] = [Peso[contar] for Peso in peso_acoes]

df = pd.DataFrame(carteira)

colunas = ['Retorno', 'Volatilidade', 'Sharpe Ratio'] + [acao+' Peso' for acao in acoes]
df = df[colunas]

#Gráfico
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatilidade', y='Retorno', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.xlabel('Volatilidade')
plt.ylabel('Retorno Esperado')
plt.title('Fronteira Eficiente de Markowitz')
plt.show()



![Image](https://user-images.githubusercontent.com/67772460/198442643-f38771f1-6f79-4c26-991b-5d38a2ad80ec.png)



#Identificar melhor sharpe e menor variancia
menor_volatilidade = df['Volatilidade'].min()
maior_sharpe = df['Sharpe Ratio'].max()

carteira_sharpe = df.loc[df['Sharpe Ratio'] == maior_sharpe]
carteira_min_variancia = df.loc[df['Volatilidade'] == menor_volatilidade]

#Novo Gráfico
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatilidade', y='Retorno', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=carteira_sharpe['Volatilidade'], y=carteira_sharpe['Retorno'], c='red', marker='o', s=200)
plt.scatter(x=carteira_min_variancia['Volatilidade'], y=carteira_min_variancia['Retorno'], c='blue', marker='o', s=200 )
plt.xlabel('Volatilidade')
plt.ylabel('Retorno Esperado')
plt.title('Fronteira Eficiente de Markowitz')
plt.show()



![Image](https://user-images.githubusercontent.com/67772460/198442773-3bdca306-0984-4b7a-af00-438ab076c630.png)


#Comparar melhores carteiras
print ("Essa é a carteira de Mínima Variância:", '\n', carteira_min_variancia.T)
print ('\n')
print ("Essa é a carteira com maior Sharpe Ratio:", '\n', carteira_sharpe.T)



![Image](https://user-images.githubusercontent.com/67772460/198442944-7902094f-32f5-4b37-b9c4-0ab7c85ced82.png)


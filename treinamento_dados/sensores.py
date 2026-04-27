import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import math

dados = pd.read_csv('dataset/dados_sensores_mistos.csv', sep=';')

dados = dados.drop(columns=['id_dispositivo'])

dados_num = dados[['frequencia_hz', 'consumo_watts', 'temperatura_interna']]
dados_cat = dados[['codigo_regiao', 'nivel_prioridade', 'status_operacional', 'versao_firmware']]

scaler = MinMaxScaler()
normalizador = scaler.fit(dados_num)

pickle.dump(normalizador, open('normalizador_sensores.pkl', 'wb'))

dados_num_norm = pd.DataFrame(
    normalizador.transform(dados_num), 
    columns=dados_num.columns,
    index=dados_cat.index # Garantir que os índices sejam os mesmos para a junção posterior
)

dados_cat_norm = pd.get_dummies(dados_cat, dtype=int)

dados_norm = dados_num_norm.join(dados_cat_norm)
print("Dados normalizados:\n", dados_norm)

pickle.dump(dados_norm.columns, open('colunas_sensores.pkl', 'wb'))

# Hiperparametrização
distorcoes = []
K = range(1, dados.shape[0])

for i in K: 
    cluster_model = KMeans(n_clusters = i, random_state=42)
    cluster_model.fit(dados_norm)

    distorcoes.append(sum(np.min(
            cdist(dados_norm, cluster_model.cluster_centers_,
                  'euclidean'), axis=1)/dados_norm.shape[0]))
    
x0 = K[0]
y0 = distorcoes[0]
x1 = K[-1]
y1 = distorcoes[-1]

distancia = []

for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    distancia.append(abs((y1-y0)*x - (x1-x0)*y + x1*y0 - y1*x0) / 
                     math.sqrt((y1-y0)**2 + (x1-x0)**2))
    
numero_clusters_otimo = K[distancia.index(np.max(distancia))]
print("Número ótimo de clusters:", numero_clusters_otimo)

cluster_model = KMeans(
    n_clusters = numero_clusters_otimo,
    random_state=42).fit(dados_norm)

print("Centros dos clusters:\n", cluster_model.cluster_centers_)

pickle.dump(cluster_model, open('modelo_clusters_sensores.pkl', 'wb'))
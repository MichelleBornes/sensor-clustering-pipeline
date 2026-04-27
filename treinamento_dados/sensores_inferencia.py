import pandas as pd
import pickle

# 1. Carregar objetos salvos
columns_name = pickle.load(open('objetos/colunas_sensores.pkl', 'rb'))
normalizador = pickle.load(open('objetos/normalizador_sensores.pkl', 'rb'))
cluster_model = pickle.load(open('objetos/modelo_clusters_sensores.pkl', 'rb'))

# 2. Definir o novo sensor (separando dados numéricos e categóricos)
novos_dados_num = [[59.8, 450.5, 82.3]]
novos_dados_cat = {'codigo_regiao': 202, 
                   'nivel_prioridade': 3, 
                   'status_operacional': ['0'], 
                   'versao_firmware': ['v2.1']}

# 3. Processar Numéricos
# O normalizador espera exatamente 3 colunas que ele conheceu no fit()
novo_sensor_norm = normalizador.transform(novos_dados_num)
df_num = pd.DataFrame(novo_sensor_norm, columns=['frequencia_hz', 'consumo_watts', 'temperatura_interna'])

# 4. Processar Categóricos (Dummies)
df_cat = pd.get_dummies(pd.DataFrame(novos_dados_cat), dtype=int)

# 5. Juntar e alinhar com as colunas do modelo
# Criar um DataFrame vazio com as colunas do modelo
sensor_final = pd.DataFrame(columns=columns_name)

# Justar o dado atual para ter as mesmas colunas do modelo
sensor_final = pd.concat([sensor_final, df_num, df_cat], axis=0)

# Preencher com 0 as colunas de categorias que não existem nesse sensor específico
sensor_final = sensor_final.fillna(0)

sensor_final = sensor_final[columns_name]  # Reordenar as colunas para garantir a ordem correta

# 6. Fazer a predição do cluster
sensor_final = cluster_model.predict(sensor_final)
print("O novo sensor pertence ao cluster:", sensor_final[0])

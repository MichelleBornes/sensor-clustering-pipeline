import pickle 
import pandas as pd

# Carregar objetos salvos
cluster_model = pickle.load(open('objetos/modelo_clusters_sensores.pkl', 'rb'))
normalizador = pickle.load(open('objetos/normalizador_sensores.pkl', 'rb'))
columns_name = pickle.load(open('objetos/colunas_sensores.pkl', 'rb'))

df = pd.DataFrame(cluster_model.cluster_centers_, columns=columns_name)

colunas_numericas =['frequencia_hz', 'consumo_watts', 'temperatura_interna']
colunas_cat = [col for col in columns_name if col not in colunas_numericas]


atributos_num_desnorm = pd.DataFrame(
    normalizador.inverse_transform(df[colunas_numericas]),
    columns=colunas_numericas
)

class_df = df[colunas_cat].round().astype(int) # Arredondar e converter para inteiro

cluster = atributos_num_desnorm.join(class_df)
print("Dados desnormalizados com sucesso:\n", cluster)
# Sensor Clustering Pipeline

Pipeline completo de **reprocessamento, clusterização e análise de dados de sensores**, utilizando técnicas de Machine Learning para identificar padrões e segmentar dispositivos automaticamente.

---

## Visão Geral

Este projeto implementa um fluxo completo de dados que inclui:

* Pré-processamento de dados numéricos e categóricos
* Normalização com MinMaxScaler
* Codificação de variáveis categóricas (One-Hot Encoding)
* Determinação automática do número ideal de clusters
* Clusterização com K-Means
* Persistência de modelos e transformações
* Predição de novos sensores
* Interpretação dos clusters com dados desnormalizados

---

## Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* SciPy
* Pickle

---

## Estrutura do Projeto

```
sensor-clustering-pipeline/
│
├── dataset/
│   └── dados_sensores_mistos.csv
│
├── objetos/
│   ├── modelo_clusters_sensores.pkl
│   ├── normalizador_sensores.pkl
│   └── colunas_sensores.pkl
│
├── treinamento_dados/
│   ├── sensores.py
│   ├── sensores_inferencia.py
│   └── sensores_desnormalizador.py
│
└── README.md
```

---

## Pipeline de Dados

### 1. Pré-processamento

* Remoção de colunas irrelevantes (`id_dispositivo`)
* Separação entre:

  * dados numéricos
  * dados categóricos

### 2. Transformações

* Normalização com `MinMaxScaler`
* Conversão de categorias com `get_dummies`

### 3. Clusterização

* Algoritmo: **K-Means**
* Determinação automática de K usando método do cotovelo com distância geométrica

### 4. Persistência

* Salvamento com `pickle`:

  * modelo treinado
  * normalizador
  * estrutura de colunas

---

## Predição de Novos Sensores

O pipeline permite classificar novos sensores:

```python
novo_sensor = [[59.8, 450.5, 82.3]]
cluster = cluster_model.predict(sensor_final)
```

Resultado: identificação automática do cluster ao qual o sensor pertence.

---

## Interpretação dos Clusters

Os centros dos clusters são:

* Desnormalizados
* Convertidos para valores reais
* Interpretados com base nas categorias

Isso permite entender o perfil de cada grupo de sensores.

---

## Como Executar

### 1. Clonar o repositório

```
git clone https://github.com/MichelleBornes/sensor-clustering-pipeline.git
cd sensor-clustering-pipeline
```

### 2. Instalar dependências

```
pip install pandas numpy scikit-learn scipy
```

### 3. Executar o pipeline

```
python treinamento_dados/sensores.py
```

---

## Autora

Desenvolvido por **Michelle Bornes**

---

## Observação

Projeto realizado para fins educacionais.

---

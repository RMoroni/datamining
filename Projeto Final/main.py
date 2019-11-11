import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
import skfuzzy as fuzz
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

dataset_path = '/home/rodrigo/Documents/dados_abertos_urbs/'
#dataset_path = '/home/viviane/Documents/dados_abertos_urbs/'
#dataset_path = 'C:'

# classe para possibilitar a customização de transformador para variáveis texto
class TextSelector(BaseEstimator, TransformerMixin):
    """
    Transformador que seleciona uma coluna do dataset para executar transformações adicionais
    Usado para colunas texto
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]

# classe para possibilitar a customização de transformador para variáveis numéricas
class NumberSelector(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform additional transformations on
    Use on numeric columns in the data
    """

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

def load_dataset(skiprows=0, nrows=None):
    entrada = pd.read_csv(dataset_path + 'entrada_cartao.csv', skiprows=skiprows, nrows=nrows, encoding='latin-1')
    cartao = pd.read_csv(dataset_path + 'cartao.csv', skiprows=skiprows, nrows=nrows)
    bairro = pd.read_csv(dataset_path + 'bairro.csv')
    full_data = [entrada, cartao, bairro]
    return full_data

def data_map(dataset):
    #cluster_map = {'X':0, 'Y':1, 'XY':2, 'L':3}
    #data['cluster'] = data['cluster'].map(cluster_map)
    return data

def all_plots(dataset):
    pass

def criaDadosCsv():

    # cria conexão com o banco de dados
    engine = create_engine('mysql://root:@localhost/projeto_grafos_multicamadas')
    conn = engine.connect()

    # busca as features das entradas de cartões
    sql_select_query = "SELECT e.NUMEROCARTAO, e.HORAUTILIZACAO, e.BAIRRO, b.CLASSE, e.SEXO FROM entrada_cartao AS e JOIN bairro AS b ON e.BAIRRO = b.NOME WHERE e.FLAGCARTAO = 1"
    entradas = pd.read_sql_query(sql_select_query, conn)

    # busca variável que será predita
    regionais = []
    for i, cartao in enumerate(entradas['NUMEROCARTAO']):
        if entradas['HORAUTILIZACAO'][i] < 10:
            sql_select_query = "SELECT b.REGIONAL FROM cartao AS c JOIN bairro AS b ON c.CODIGOBAIRRODESTINO = b.CODIGO WHERE c.NUMERO = " + cartao
            regional = pd.read_sql_query(sql_select_query, conn)
            regionais.append(regional['REGIONAL'][0])
        else:
            sql_select_query = "SELECT b.REGIONAL FROM cartao AS c JOIN bairro AS b ON c.CODIGOBAIRROMORADIA = b.CODIGO WHERE c.NUMERO = " + cartao
            regional = pd.read_sql_query(sql_select_query, conn)
            regionais.append(regional['REGIONAL'][0])

    entradas['REGIONAL'] = regionais
    entradas.to_csv('entradas.csv')

    return entradas

def calculaCorrelacao(var_1, var_2):

    print('Correlação de Pearson: ',var_1.corr(var_2))

    print('Correlação de Spearman: ', var_1.corr(var_2, method="spearman"))

if __name__ == "__main__":
    print('Carregando os dados...')
    dataset = load_dataset()

    # entradas = alimentaDataframe()
    # print(entradas.head())

    # lê o dataset
    entradas = pd.read_csv('entradas.csv', encoding='utf-8')
    # seta a coluna como index
    entradas.set_index('INDEX', inplace=True)

    # binariza a classe de predição e as features categóricas para calcular as correlações
    label_encoder = LabelEncoder()
    regional_binarizada = label_encoder.fit_transform(entradas['REGIONAL'])
    entradas['REGIONALBINARIZADA'] = regional_binarizada
    bairro_binarizado = label_encoder.fit_transform(entradas['BAIRRO'])
    entradas['BAIRROBINARIZADO'] = bairro_binarizado

    # calculaCorrelacao(entradas['HORAUTILIZACAO'], entradas['REGIONALBINARIZADA'])
    '''Correlação de Pearson:  -0.05839568006612773
    Correlação de Spearman:  -0.05537727124768886'''

    # calculaCorrelacao(entradas['CLASSE'], entradas['REGIONALBINARIZADA'])
    '''Correlação de Pearson:  0.13401692823046568
    Correlação de Spearman:  0.1516167437961979'''

    # calculaCorrelacao(entradas['BAIRROBINARIZADO'], entradas['REGIONALBINARIZADA'])
    '''Correlação de Pearson:  0.10576365731450334
    Correlação de Spearman:  0.09956378855415446'''

    # recebe os nomes das features e da variável alvo
    features = ['HORAUTILIZACAO', 'BAIRROBINARIZADO', 'CLASSE', 'SEXO']
    numeric_features = ['HORAUTILIZACAO', 'CLASSE', 'BAIRROBINARIZADO']
    target = 'REGIONAL'

    # separa os dados em um conjunto de treino e outro de teste
    X_train, X_test, y_train, y_test = train_test_split(entradas[features], entradas[target], test_size=0.33,
                                                        random_state=42)

    #########################PIPELINE#########################

    # cria mini pipelines, um para cada transformador

    # seleciona e binariza variável sexo
    '''sexo = Pipeline([
        ('selector', TextSelector(key='SEXO')),
        ('one-hot encoder', OneHotEncoder())
    ])'''

    # seleciona a variável bairro
    bairro = Pipeline([
        ('selector', NumberSelector(key='BAIRROBINARIZADO'))
    ])

    # seleciona variável hora
    hora = Pipeline([
        ('selector', NumberSelector(key='HORAUTILIZACAO'))
    ])

    # seleciona variável classe
    classe = Pipeline([
        ('selector', NumberSelector(key='CLASSE'))
    ])

    # junta os mini pipelines
    '''feats = FeatureUnion([('sexo', sexo),
                          ('bairro', bairro),
                          ('hora', hora),
                          ('classe', classe)])'''

    feats = FeatureUnion([('bairro', bairro),
                          ('hora', hora),
                          ('classe', classe)])

    # aplica as transformações nos dados de treino
    feature_processing = Pipeline([('feats', feats)])
    feature_processing.fit_transform(X_train)

    # cria o pipeline com os transformadores e o classificador escolhido
    '''pipeline = Pipeline([
        ('features', feats),
        ('classifier', DecisionTreeClassifier(max_depth=3, random_state=0))
    ])'''

    pipeline = Pipeline([
        ('features', feats),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # cria o modelo
    pipeline.fit(X_train, y_train)

    # aplica os dados de teste no modelo
    preds = pipeline.predict(X_test)

    # valida o modelo (média de acertos)
    print(np.mean(preds == y_test))
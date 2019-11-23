import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LinearRegression
#from sklearn.cluster import KMeans
#from sklearn.metrics import mean_squared_error
#from sklearn.neural_network import MLPRegressor
#import skfuzzy as fuzz
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
from datetime import datetime

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
    entrada = pd.read_csv(dataset_path + 'entrada_cartao.csv', skiprows=skiprows, nrows=nrows, encoding='latin-1', sep=';')
    cartao = pd.read_csv(dataset_path + 'cartao.csv', skiprows=skiprows, nrows=nrows)
    bairro = pd.read_csv(dataset_path + 'bairro.csv')
    full_data = [entrada, cartao, bairro]
    return full_data

def data_treat(dataset):
    # Tratamento das data de nascimento para idade
    dataset['DATANASCIMENTO'].fillna('01/01/20', inplace=True)  # Isso vira 01/01/2020
    # Gera uma lista contendo as idades ao invés da data de nsc
    idade_list = [(2019 - datetime.strptime(dt_nsc, '%d/%m/%y').year) for dt_nsc in dataset['DATANASCIMENTO'].tolist()]
    # Cria nova coluna (idade) no DF
    dataset['IDADE'] = idade_list
    # Ninguem tem -1 de idade
    dataset.loc[(dataset['IDADE'] == -1), 'IDADE'] = 38  # 38 é a idade média
    dataset.loc[(dataset['IDADE'] >= -30) & (dataset['IDADE'] < -20), 'IDADE'] = 84
    dataset.loc[(dataset['IDADE'] >= -40) & (dataset['IDADE'] < -30), 'IDADE'] = 74
    dataset.loc[(dataset['IDADE'] >= -50) & (dataset['IDADE'] < -40), 'IDADE'] = 64

    # Tratamento da data/hora de embarque para dias da semana ("Monday is 0 and Sunday is 6")
    dia_semana_embarque_list = [(datetime.strptime(dt_emb, '%Y-%m-%d').weekday()) for dt_emb in dataset[0]['DATAUTILIZACAO']]
    dataset[0]['DIASEMANA'] = dia_semana_embarque_list

    # Dia Útil
    # Não achei um jeito automático então por enquanto é colocar os feriados aqui :(
    feriados = ['2018-09-07', '2018-10-12', '2018-11-15', '2018-11-02']
    dia_util = []
    for dia_semana, dt_emb in zip(dia_semana_embarque_list, dataset[0]['DATAUTILIZACAO']):
        # Entre sábado e domingo ou em um dos feriados ñ é dia útil
        if ((dia_semana >= 5 and dia_semana <= 6) or dt_emb in feriados):
            dia_util.append(False)
            print(str(dia_semana) + ' - ' + dt_emb)
        else:
            dia_util.append(True)
    dataset[0]['DIA_UTIL'] = dia_util

    # Faixa Etária
    faixa_etaria = []
    for idade in dataset[0]['IDADE']:
        if idade > 0 and idade < 20:
            faixa_etaria.append('JOVEM')
        elif idade >= 20 and idade < 60:
            faixa_etaria.append('ADULTO')
        elif idade >= 60:
            faixa_etaria.append('IDOSO')
        else:
            faixa_etaria.append('ERRO')
    dataset[0]['FAIXAETARIA'] = faixa_etaria

    # Binarização da coluna Sexo
    sexoBinarizado = []
    for sexo in dataset['SEXO']:
        if sexo == 'M':
            sexoBinarizado.append(0)
        else:
            sexoBinarizado.append(1)
    dataset['SEXOBINARIZADO'] = sexoBinarizado

    # Tratamento da data e hora para timestamp
    hora_list = [str(hora) for hora in dataset['HORAUTILIZACAO']]
    timestamp_list = [int(time.mktime(datetime.strptime(dt + ' ' + hora_list[i], "%Y-%m-%d %H").timetuple())) for i, dt
                      in enumerate(dataset['DATAUTILIZACAO'])]
    dataset['TIMESTAMP'] = timestamp_list

    # Binarização das features categóricas utilizadas na predição
    label_encoder = LabelEncoder()
    bairro_binarizado = label_encoder.fit_transform(entradas['BAIRRO'])
    entradas['BAIRROBINARIZADO'] = bairro_binarizado
    regional_binarizada = label_encoder.fit_transform(entradas['REGIONAL'])
    entradas['REGIONALBINARIZADA'] = regional_binarizada
    cod_linha_binarizado = label_encoder.fit_transform(entradas['CODLINHA'])
    entradas['CODLINHABINARIZADO'] = cod_linha_binarizado

    print(dataset[0])
    return dataset

def data_map(dataset):
    #cluster_map = {'X':0, 'Y':1, 'XY':2, 'L':3}
    #data['cluster'] = data['cluster'].map(cluster_map)
    return dataset

def show_plots(dataset):
    entrada = dataset[0]
    cartao = dataset[1]
    bairros = dataset[2]

    # Box Plot da renda média dos bairros
    plt.title('Renda Média dos Bairros')
    plt.boxplot(bairros['RENDAMEDIAMENSAL'])
    plt.show()

    # Pie das origens
    plt.title('Bairros de Origem')
    plt.pie(cartao['CODIGOBAIRROORIGEM'].value_counts(), autopct='%1.1f%%')
    plt.show()

    # Pie dos destino
    plt.title('Bairros de Destino')
    plt.pie(cartao['CODIGOBAIRRODESTINO'].value_counts(), autopct='%1.1f%%')
    plt.show()

    # Uso de acordo com sexo
    plt.title('Utilização por sexo')
    plt.pie(entrada['SEXO'].value_counts(), labels=['F', 'M'], autopct='%1.1f%%')
    plt.show()

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

if __name__ == "__main__":
    print('Carregando os dados...')
    dataset = load_dataset()
    print('Tratamento dos dados...')
    dataset = data_treat(dataset)
    #print('Plotagem...')
    #show_plots(dataset)
    exit(0)

    # entradas = alimentaDataframe()
    # print(entradas.head())

    # lê o dataset
    entradas = pd.read_csv(dataset_path + 'entradas.csv', encoding='utf-8')

    plt.title('Regionais')
    plt.pie(entradas['REGIONAL'].value_counts(), labels=['Matriz', 'Boa Vista', 'St Felicidade', 'CIC', 'Fazendinha', 'Pinheirinho', 'Cajuru', 'Boqueirao', 'Tatu', 'Portao', 'Bairro Novo'],autopct='%1.1f%%')
    plt.show()

    print(entradas['REGIONAL'].value_counts())

    # seta a coluna como index
    entradas.set_index('INDEX', inplace=True)

    # cria as features
    entradas = data_treat(entradas)

    # recebe os nomes das features e da variável alvo
    features = ['DIASEMANA', 'HORAUTILIZACAO', 'CODLINHABINARIZADO', 'REGIONALBINARIZADA', 'CLASSE', 'SEXOBINARIZADO', 'IDADE', 'BAIRROBINARIZADO', 'TIMESTAMP']
    target = 'REGIONALDESTINO'

    # separa os dados em um conjunto de treino e outro de teste
    #X_train, X_test, y_train, y_test = train_test_split(entradas[features], entradas[target], test_size=0.33, random_state=42)

    # NÃO ENCONTREI UMA FORMA AUTOMÁTICA DE SEPARAR OS DADOS EM TREINO E TESTE ESTABELECENDO DIFERENTES PROPORÇÕES PARA CADA
    # CONJUNTO DE CLASSES DE PREDIÇÃO, POR ISSO FIZ MANUAL MESMO
    # filtra as entradas por regional de destino
    bairro_novo = entradas[entradas['REGIONALDESTINO'] == "Bairro Novo"]
    boa_vista = entradas[entradas['REGIONALDESTINO'] == "Boa Vista"]
    cajuru = entradas[entradas['REGIONALDESTINO'] == "Cajuru"]
    boqueirao = entradas[entradas['REGIONALDESTINO'] == "Boqueirao"]
    cic = entradas[entradas['REGIONALDESTINO'] == "CIC"]
    fazendinha_portao = entradas[entradas['REGIONALDESTINO'] == "Fazendinha/Portao"]
    matriz = entradas[entradas['REGIONALDESTINO'] == "Matriz"]
    pinheirinho = entradas[entradas['REGIONALDESTINO'] == "Pinheirinho"]
    santa_felicidade = entradas[entradas['REGIONALDESTINO'] == "Santa Felicidade"]
    tatuquara = entradas[entradas['REGIONALDESTINO'] == "Tatuquara"]

    # separa os dados em um conjunto de treino e outro de teste com proporções diferentes para cada classe de predição
    X_train_matriz, X_test_matriz, y_train_matriz, y_test_matriz = train_test_split(matriz[features], matriz[target], train_size=0.2, random_state=42)
    X_train_santa, X_test_santa, y_train_santa, y_test_santa = train_test_split(santa_felicidade[features], santa_felicidade[target], train_size=0.3, random_state=42)
    X_train_boa, X_test_boa, y_train_boa, y_test_boa = train_test_split(boa_vista[features], boa_vista[target], train_size=0.3, random_state=42)
    X_train_cic, X_test_cic, y_train_cic, y_test_cic = train_test_split(cic[features], cic[target], train_size=0.3, random_state=42)
    X_train_fazendinha, X_test_fazendinha, y_train_fazendinha, y_test_fazendinha = train_test_split(fazendinha_portao[features], fazendinha_portao[target], train_size=0.4, random_state=42)
    X_train_pinheirinho, X_test_pinheirinho, y_train_pinheirinho, y_test_pinheirinho = train_test_split(pinheirinho[features], pinheirinho[target], train_size=0.5, random_state=42)
    X_train_cajuru, X_test_cajuru, y_train_cajuru, y_test_cajuru = train_test_split(cajuru[features], cajuru[target], train_size=0.6, random_state=42)
    X_train_boqueirao, X_test_boqueirao, y_train_boqueirao, y_test_boqueirao = train_test_split(boqueirao[features], boqueirao[target], train_size=0.6, random_state=42)
    X_train_tatuquara, X_test_tatuquara, y_train_tatuquara, y_test_tatuquara = train_test_split(tatuquara[features], tatuquara[target], train_size=0.8, random_state=42)
    X_train_bairro, X_test_bairro, y_train_bairro, y_test_bairro = train_test_split(bairro_novo[features], bairro_novo[target], train_size=0.8, random_state=42)

    # junta os subconjuntos de treino e teste
    X_train = pd.concat([X_train_matriz, X_train_santa, X_train_boa, X_train_cic, X_train_fazendinha, X_train_pinheirinho, X_train_cajuru, X_train_boqueirao, X_train_tatuquara, X_train_bairro])
    X_test = pd.concat([X_test_matriz, X_test_santa, X_test_boa, X_test_cic, X_test_fazendinha, X_test_pinheirinho, X_test_cajuru, X_test_boqueirao, X_test_tatuquara, X_test_bairro])
    y_train = pd.concat([y_train_matriz, y_train_santa, y_train_boa, y_train_cic, y_train_fazendinha, y_train_pinheirinho, y_train_cajuru, y_train_boqueirao, y_train_tatuquara, y_train_bairro])
    y_test = pd.concat([y_test_matriz, y_test_santa, y_test_boa, y_test_cic, y_test_fazendinha, y_test_pinheirinho, y_test_cajuru, y_test_boqueirao, y_test_tatuquara, y_test_bairro])

    #########################PIPELINE#########################

    # cria mini pipelines, um para cada transformador

    # seleciona e binariza variável sexo
    sexo = Pipeline([
        ('selector', NumberSelector(key='SEXOBINARIZADO'))
    ])

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

    dia_semana = Pipeline([
        ('selector', NumberSelector(key='DIASEMANA'))
    ])

    cod_linha = Pipeline([
        ('selector', NumberSelector(key='CODLINHABINARIZADO'))
    ])

    regional = Pipeline([
        ('selector', NumberSelector(key='REGIONALBINARIZADA'))
    ])

    idade = Pipeline([
        ('selector', NumberSelector(key='IDADE'))
    ])

    timestamp = Pipeline([
        ('selector', NumberSelector(key='TIMESTAMP'))
    ])

    # junta os mini pipelines
    feats = FeatureUnion([('sexo', sexo),
                          ('hora', hora),
                          ('codLinha', cod_linha),
                          ('bairro', bairro),
                          ('idade', idade)])

    # aplica as transformações nos dados de treino
    feature_processing = Pipeline([('feats', feats)])
    feature_processing.fit_transform(X_train)

    # cria o pipeline com os transformadores e o classificador escolhido
    pipeline = Pipeline([
        ('features', feats),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # cria o modelo
    pipeline.fit(X_train, y_train)

    # aplica os dados de teste no modelo
    preds = pipeline.predict(X_test)

    # imprime os labels para a matriz
    print("Bairro Novo", "Boa Vista", "Cajuru", "Boqueirao", "CIC", "Fazendinha/Portao", "Matriz", "Pinheirinho", "Santa Felicidade", "Tatuquara")

    # valida o modelo
    # matriz de confusão
    print(confusion_matrix(y_test, preds, labels=["Bairro Novo", "Boa Vista", "Cajuru", "Boqueirao", "CIC", "Fazendinha/Portao", "Matriz", "Pinheirinho", "Santa Felicidade", "Tatuquara"]))

    # acurácia
    print('Acurácia -> ', accuracy_score(y_test, preds))

    # recall
    print('Recall -> ', recall_score(y_test, preds, average='macro'))

    '''
    MELHOR ATÉ AGORA:
    Acurácia ->  0.6316002238945208
    Recall ->  0.6458925915711106
    '''

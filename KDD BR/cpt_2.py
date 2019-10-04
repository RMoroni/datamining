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

dataset_path = '/home/rodrigo/Documents/kddbr-2019/public/'
#dataset_path = '/home/viviane/Documents/kddbr-2019/public/'
#dataset_path = 'C:'

def load_dataset(skiprows, nrows):
    #abre o csv de treino e teste no caminho específicado
    train = pd.read_csv(dataset_path + 'training_dataset.csv', skiprows=skiprows, nrows=nrows)
    test = pd.read_csv(dataset_path + 'test_dataset.csv', skiprows=skiprows, nrows=nrows)
    score = pd.read_csv(dataset_path + 'training_data_labels.csv')
 
    #transforma em um único array e retorna
    full_data = [train, test, score]
    return full_data
 
def data_map(data):
    #Mapeia o valor de uma string para um int
    cluster_map = {'X':0, 'Y':1, 'XY':2, 'L':3}
    data['cluster'] = data['cluster'].map(cluster_map)
    return data

def drop_data(data):
    #retira as duas colunas, basicamentes elas não fazem diferença
    return data.drop(labels=['datapointID', 'sampletype'], axis=1)

def data_treat(dataset):
    #por enquanto vazio, mas aqui pode ser feito tratamentos
    dataset[0] = data_map(dataset[0])
    dataset[1] = data_map(dataset[1])
    dataset[0] = drop_data(dataset[0])
    dataset[1] = drop_data(dataset[1])
    return dataset

def show_plots(data):
    train = data[0][0:3840] #seleciona uma parte pra não ficar muito tempo no for...
    score = data[2]
 
    #Boxplot da silhu
    #plt.boxplot(train['silhouette'])
    #plt.show()

    #boxplot do score
    plt.boxplot(score['score'])
    plt.title('Box Plot do Score (completo)')
    plt.show()
 
    #Histograma do score
    plt.hist(score['score'])
    plt.title('Histograma do Score (completo)')
    plt.show()

    #esse trecho ajuda a mostrar as diferenças entre cada amostra
    for id, sample in train.groupby('scatterplotID'):        
        #gráfico de X e Y com labels (seguindo as cores)
        plot_score = score[score['scatterplotID'] == id].score.values[0]
        silhouette = train[train['scatterplotID'] == id].silhouette
        print(silhouette.mean())
        colors = ['red', 'green', 'blue', 'purple']
        plt.scatter(sample['signalX'], sample['signalY'], c=sample['cluster'], cmap=matplotlib.colors.ListedColormap(colors))
        plt.title('scatterplotID:' + str(id) + ' - Score:' + str(plot_score))
        plt.show()
        plt.hist(silhouette)
        plt.show()

def frequency_clusters(dataset):
    #filtra amostras com mais alto e com mais baixo score
    alto_score = dataset[2][dataset[2]['score'] >= 0.9]
    baixo_score = dataset[2][dataset[2]['score'] <= 0.1]

    #recupera os osbjetos de cada amostra filtrada anteriomente
    frames_alto_score = dataset[0][dataset[0]['scatterplotID'].isin(alto_score['scatterplotID'])]
    frames_baixo_score = dataset[0][dataset[0]['scatterplotID'].isin(baixo_score['scatterplotID'])]
    
    '''
    frames_alto_score = []
    for id_amostra in alto_score['scatterplotID']:
        frames_alto_score.append(dataset[0][dataset[0]['scatterplotID'] == id_amostra])

    frames_baixo_score = []
    for id_amostra in baixo_score['scatterplotID']:
        frames_baixo_score.append(dataset[0][dataset[0]['scatterplotID'] == id_amostra])
    '''
    
    #concatena os frames recuperados
   # dataset_alto_score = pd.concat(frames_alto_score)
   # dataset_baixo_score = pd.concat(frames_baixo_score)

    #plota o histograma para verificar se existem diferenças nas frequencias de clusters em cada caso
    plt.hist(frames_alto_score['cluster'])
    plt.title('Frequência por Cluster (score > 0.9)')
    plt.show()

    plt.hist(frames_baixo_score['cluster'])
    plt.title('Frequência por Cluster (score < 0.1)')
    plt.show()

def linear_regression(dataset):
    #regressão linear simples, a feature atual é só um exemplo...
    train = dataset[0][0:]
    score = dataset[2]

    #retiro os marcados como L
    train = train[train['cluster'] != 3]

    #no X será a feature que tem correlação com score e no Y o score
    x = []
    y = []

    feat = 0.0

    #regressão linear da média da silhouette de cada plot com o score
    for id, sample in train.groupby('scatterplotID'):
        feat = sample.silhouette.median()*0.1
        feat = feat - sample.silhouette.std()
        x.append(feat) #feature
        y.append(score[score['scatterplotID'] == id].score.values[0]) #score do scatterplot
        feat = 0.0

    x = np.array(x).reshape((-1, 1))
    y = np.array(y)

    model = LinearRegression().fit(x, y) #gera o modelo de regressão linear
    r_sq = model.score(x, y) #calcular a taxa de correlação
    print(r_sq)

    y_pred = model.predict(x)
    score['train_predict'] = y_pred
    train.to_csv(dataset_path + 'train_predict.csv')
    #print(y_pred)

def linear_regression_plot(dataset):
    #ideia era plotar um gráfico (feature X score) para buscar uma feature que pudesse ser aplicado a regressão linear
    train = dataset[0][0:]
    score = dataset[2]

    #retiro os marcados como L
    train = train[train['cluster'] != 3]

    #train = train[train['scatterplotID'] == 33285]
    x = []
    y = []

    feat = 1.0
    #regressão linear da média da silhouette de cada plot com o score
    for id, sample in train.groupby('scatterplotID'):
        feat = feat - (abs(sample.silhouette.mean()-sample.silhouette.median()) + sample.silhouette.std())
        x.append(feat)
        y.append(score[score['scatterplotID'] == id].score.values[0])
        feat = 1.0

    #basicamente, se a feat estiver 'boa', os dados poderão ser aproximados por uma reta
    plt.scatter(x, y)
    plt.show()

def silhouette_statistic(dataset):
    #salva algumas features a partir da silhueta, a ideia era encontra alguma correlação com o score
    #correlçao não foi encontrada, mas fica como exemplo de que deu errado
    train = dataset[0][0:]
    score = dataset[2]

    #retira com label L
    train = train[train['cluster'] != 3]
    
    mean = []
    median = []
    std = []
    qt1 = []
    qt3 = []
    scr = []

    for id, sample in train.groupby('scatterplotID'):
        mean.append(sample.silhouette.mean())
        median.append(sample.silhouette.median())
        std.append(sample.silhouette.std())
        qt1.append(sample.silhouette.quantile(q=0.25))
        qt3.append(sample.silhouette.quantile(q=0.75))
        scr.append(score[score['scatterplotID'] == id].score.values[0])

    dict = {'score':scr, 'mean':mean, 'median':median, 'quartil 1':qt1, 'quartil 3':qt3, 'std':std}

    df = pd.DataFrame(dict)
    df.to_csv(dataset_path + 'silhouette_statistic.csv')

def k_means(dataset):
    #kmeans, ideia era agrupar e descobrir quais pontos era díficeis de rotular (e em seguida estimar o score)
    #como a clusterização ficou muito longe da original, a ideia com kmeans foi abondonada, mas seguie para o fuzzy
    train = dataset[0][0:3840]
    score = dataset[2]
    train = train[train['cluster'] != 3]
    
    acurr = []
    X = []
    n_clusters = 0
    for id, sample in train.groupby('scatterplotID'):

        #separa os clusters para descobrir o centro de cada um
        cluster_x = sample[sample['cluster'] == 0]
        cluster_y = sample[sample['cluster'] == 1]
        cluster_xy = sample[sample['cluster'] == 2]    

        init = [] #centroide
                
        #para descobrir quantos clusters será utilizado no kmeans
        #certeza tem jeito mais simples de fazer isso, mas funcionou então ok...
        if len(cluster_x) > 0:
            n_clusters = n_clusters + 1
            init.append([cluster_x.signalX.median(), cluster_x.signalY.median()])
        if len(cluster_y) > 0:
            n_clusters = n_clusters + 1            
            init.append([cluster_y.signalX.median(), cluster_y.signalY.median()])
        if len(cluster_xy) > 0:
            n_clusters = n_clusters + 1            
            init.append([cluster_xy.signalX.median(), cluster_xy.signalY.median()])        
    
        init = np.array(init)

        #como o centroide já está no lugar 'certo', n_init=1, pra não tentar alterar posição do centroide
        kmeans = KMeans(n_clusters=n_clusters, init=init, n_jobs=-1, n_init=1, max_iter=300)

        #percorre todas linhas do scatter e adiciona no vetor
        for _, row in sample.iterrows():        
            X.append([row['signalX'], row['signalY']])

        #salvo o score do plot
        plot_score = score[score['scatterplotID'] == id].score.values[0]
        
        #plt.title(f"Score: {plot_score}") #se for python 3.6 ou superior, pode rodar dessa forma
        plt.title('Score:' + str(plot_score))
        
        #aquele padrãozinho
        X = np.array(X)
        y_kmeans = kmeans.fit_predict(X)
        
        #comparar a rotulagem do kmeans com original
        count = 0
        hit = 0
        for _, row in sample.iterrows():
            if row['cluster'] == y_kmeans[count]:
                hit = hit + 1
            count = count + 1
        print('Acerto:' + str(hit/len(y_kmeans)))
        acurr.append(hit/len(y_kmeans))

        #plotagem do kmeans
        plt.scatter(X[:,0], X[:,1], c=y_kmeans, cmap='viridis')
        plt.show()
        X = [] #zera vetor
        n_clusters = 0 #zera clusters

    return acurr

def fuzzy(dataset):
    # para armazenar o fpc de cada amostra
    FPC = []
    # para armazenar o erro fuzzy de cada amostra
    ERRO = []
    # para armazenar a quantidade de objetos L de cada amostra
    L = []

    # para cada amostra...
    for id, amostra in data.groupby('scatterplotID'):
        # armazena o cluster de cada objeto em uma lista
        classificacao_column = amostra['cluster']
        classificacao = []
        for classific in classificacao_column:
            classificacao.append(classific)

        # armazena o sinal X de cada objeto em uma lista
        sinal_x_column = amostra['signalX']
        sinal_x = []
        for ponto in sinal_x_column:
            sinal_x.append(ponto)

        # armazena o sinal Y de cada objeto em uma lista
        sinal_y_column = amostra['signalY']
        sinal_y = []
        for ponto in sinal_y_column:
            sinal_y.append(ponto)

        # cria um np array com as listas do sinal X e do sinal Y
        data = np.array([sinal_x, sinal_y])

        init = []  # centroide

        # separa os clusters para descobrir o centro de cada um
        cluster_x = amostra[amostra['cluster'] == 0]
        cluster_y = amostra[amostra['cluster'] == 1]
        cluster_xy = amostra[amostra['cluster'] == 2]

        # para descobrir quantos clusters existem e os centróides de cada um
        n_clusters = 0
        if len(cluster_x) > 0:
            n_clusters = n_clusters + 1
            init.append([cluster_x.signalX.median(), cluster_x.signalY.median()])
        if len(cluster_y) > 0:
            n_clusters = n_clusters + 1
            init.append([cluster_y.signalX.median(), cluster_y.signalY.median()])
        if len(cluster_xy) > 0:
            n_clusters = n_clusters + 1
            init.append([cluster_xy.signalX.median(), cluster_xy.signalY.median()])

        init = np.array(init)

        # realiza o agrupamento passando os centróides como parâmetro
        u, _, _, _, _, fpc = fuzz.cluster.cmeans_predict(data, init, 2, error=0.005, maxiter=1000)

        # refaz o vetor de classificação porque será utilizado para acessar a matriz fuzzy
        if n_clusters == 2:
            # guarda valores únicos
            val, _ = np.unique(classificacao, return_counts=True)
            val = np.sort(val)
            classificacao = np.where(classificacao == val[0], 0, classificacao)
            classificacao = np.where(classificacao == val[1], 1, classificacao)
        elif n_clusters == 1:
            val, _ = np.unique(classificacao, return_counts=True)
            classificacao = np.where(classificacao == val[0], 0, classificacao)

        # filtra os objetos mal classificados e armazena numa lista o erro de cada um
        erros = []
        if n_clusters == 3:
            for i in range(len(classificacao)):
                if u[classificacao[i]][i] < 0.45 and u[classificacao[i]][i] > 0.2:
                    erros.append(1 - u[classificacao[i]])
        elif n_clusters == 2:
            for i in range(len(classificacao)):
                if u[classificacao[i]][i] < 0.65 and u[classificacao[i]][i] > 0.3:
                    erros.append(1 - u[classificacao[i]])
        else:
            for i in range(len(classificacao)):
                if u[classificacao[i]][i] < 0.85 and u[classificacao[i]][i] > 0.4:
                    erros.append(1 - u[classificacao[i]])

        media = 0
        if len(erros) > 0:
            media = np.mean(erros)  # calcula a média dos erros
        else:
            pass

        # armazena o fpc da amostra
        FPC.append(fpc)
        # realiza o cálculo para ponderar o erro e armazena
        ERRO.append(media * (len(erros) / len(classificacao)))
        # armazena a quantidade de objetos L
        L.append(384 - len(classificacao))

    return [ERRO, L, FPC]

def submission(test_id, predict):
    #recebe dois vetores, transforma em dataframe e salva o csv
    submission_df = pd.DataFrame({'scatterplotID':test_id, 'score':predict})
    submission_df.to_csv(dataset_path + 'submission.csv', index=False)

def mlp_simples(dataset):
    #detalhe que só estou usando uma parte do treino, esse valor (38400) tem que ser compativel com valor no score(118) por causa do shape
    train = dataset[0][0:38400]
    train = train[train['cluster'] != 3]
    test = dataset[1]
    test = test[test['cluster'] != 3]
    score = dataset[2][0:118]
    
    X = [] #dados para treino
    Y = score['score'].values #o resultado que quero prever

    fuzzy_return = fuzzy(train)
    fuzzy_df = pd.DataFrame({'id':fuzzy_return[2], 'erro':fuzzy_return[1], 'fpc':fuzzy_return[0]})

    #preencho os dados de treino
    for id, sample in train.groupby('scatterplotID'):
        fuzzy_sample = fuzzy_df[fuzzy_df['id'] == id]
        X.append([sample.silhouette.std(), 
        sample.silhouette.median(),
        sample.silhouette.max(),
        sample.silhouette.min(),
        sample.signalY.max(),
        sample.signalY.min(),
        sample.signalY.mean(),
        sample.signalX.max(),
        sample.signalX.min(),
        sample.signalX.mean(),
        fuzzy_sample.erro,
        fuzzy_sample.fpc])
    
    #normalização
    X = np.array(X)
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    #treinamento: com 3 de 100 fica um resultado 'razoavel', mudando pra 1000 a convergencia é mais rápida)
    #max_iter: a convergência ocorre um pouco depois de 100, então 200 é razoável...
    mlp = MLPRegressor(hidden_layer_sizes=(1000,1000,1000), max_iter=200, verbose=True)
    mlp.fit(X, Y)

    #predição usando treino
    print(mlp.predict(X[0:10]))

    #pra salvar os testes no submission, só descomentar esse trecho
    '''
    fuzzy_return = None
    fuzzy_return = fuzzy(test)
    fuzzy_df = None
    fuzzy_df = pd.DataFrame({'id':fuzzy_return[2], 'erro':fuzzy_return[1], 'fpc':fuzzy_return[0]})
    X_test = []
    test_id = []
    for id, sample in test.groupby('scatterplotID'):
        fuzzy_sample = fuzzy_df[fuzzy_df['id'] == id]
        X_test.append([sample.silhouette.std(), 
        sample.silhouette.median(),
        sample.silhouette.max(),
        sample.silhouette.min(),
        sample.signalY.max(),
        sample.signalY.min(),
        sample.signalY.mean(),
        sample.signalX.max(),
        sample.signalX.min(),
        sample.signalX.mean(),
        fuzzy_sample.erro,
        fuzzy_sample.fpc])
        test_id.append(str(id))
    
    #normaliza os dados de teste
    X_test = np.array(X_test)
    scaler = StandardScaler()
    scaler.fit(X_test)
    X_test = scaler.transform(X_test)

    #faz a predição e salva em predict
    predict = mlp.predict(X_test)

    #envia para ser salvo no arquivo
    submission(test_id, predict)'''

def multiple_linear_regression(dataset):
    # separa os dados de treino e de teste
    train = dataset[0][0:]
    test = dataset[1][0:]
    score = dataset[2]

    # retira os objetos L
    train = train[train['cluster'] != 3]
    test = test[test['cluster'] != 3]

    # aplica a técnica fuzzy nos dados de treino e separa as features de retorno
    valores_fuzzy = fuzzy(train)
    erros_fuzzy = valores_fuzzy[0]
    l_fuzzy = valores_fuzzy[1]
    fpc_fuzzy = valores_fuzzy[2]

    # inicializa as features estatísticas e o score
    mediana = []
    desvio_padrao = []
    variancia = []
    media = []
    desvio_absoluto = []
    y = []

    i = 0
    for id, sample in train.groupby('scatterplotID'):
        # calcula e armazena as features estatísticas
        variancia.append(sample.silhouette.var())
        media.append(sample.silhouette.mean())
        mediana.append(sample.silhouette.median())
        desvio_padrao.append(sample.silhouette.std())
        desvio_absoluto.append(sample.silhouette.mad())

        # armazena os scores dos dados de treino
        y.append(score[score['scatterplotID'] == id].score.values[0])
        i += 1

    # agrupa as variaveis preditoras
    x = np.column_stack((erros_fuzzy, l_fuzzy, fpc_fuzzy, mediana, desvio_padrao, variancia, media, desvio_absoluto))
    # separa o conjunto de dados em Conjunto de Treino e Validação
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)

    y = np.array(y)
    # gera o modelo de regressão linear
    #model = LinearRegression().fit(x, y)
    model = LinearRegression().fit(X_train, y_train)

    # PREDIÇÃO
    # realiza o procedimento de criação das features para os dados de teste

    valores_fuzzy_2 = fuzzy(test)
    erros_fuzzy_2 = valores_fuzzy_2[0]
    l_fuzzy_2 = valores_fuzzy_2[1]
    fpc_fuzzy_2 = valores_fuzzy_2[2]

    mediana_2 = []
    desvio_padrao_2 = []
    variancia_2 = []
    media_2 = []
    desvio_absoluto_2 = []

    scatterploid = []
    for id_2, sample_2 in test.groupby('scatterplotID'):
        scatterploid.append(id_2)
        mediana_2.append(sample_2.silhouette.median())
        desvio_padrao_2.append(sample_2.silhouette.std())
        variancia_2.append(sample_2.silhouette.var())
        media_2.append(sample_2.silhouette.mean())
        desvio_absoluto_2.append(sample_2.silhouette.mad())

    x_2 = np.column_stack((erros_fuzzy_2, l_fuzzy_2, fpc_fuzzy_2, mediana_2, desvio_padrao_2, variancia_2, media_2, desvio_absoluto_2))

    #usa o modelo criado anteriormente para prever o score das amostras de teste
    #y_pred = model.predict(X_test)
    y_pred = model.predict(x_2)

    # calcula a taxa de correlação
    # r_sq = model.score(x, y)
    r_sq = model.score(X_train, y_train)
    print('Taxa de correlação: ', r_sq)

    # MSE Score perto de 0 é um bom modelo
    print(f"MSE score: {mean_squared_error(y_test, y_pred)}")

    # cria o csv de acordo com o modelo do kaggle
    df = pd.DataFrame()
    df['scatterplotID'] = scatterploid
    df['score'] = y_pred
    df.to_csv(dataset_path + 'submission.csv')

if __name__ == "__main__":
    
    #define parte do csv que será carregada
    #dessa forma é possível fazer o carregamento e processamento por partes
    skiprows = 0 #Pula nenhuma linha
    #skiprows = range(1,384) #Ou seja ignora as linhas de 1 a 385 (preciso da linha 0 p/ colunas)
    nrows = None #Quantas linhas serão carregadas
 
    #carrega o dataset
    dataset = load_dataset(skiprows, nrows)
 
    #trata o dataset
    dataset = data_treat(dataset)

    #rede neural (utiliza o fuzzy)
    mlp_simples(dataset)
    
    #onde serão montados os gráficos
    #show_plots(dataset)

    #mostra a frequencia dos clusters para amostras com alto score e baixo score
    #frequency_clusters(dataset)

    #plot para ajudar a montar a regressão
    #linear_regression_plot(dataset)

    #regressão linear
    #linear_regression(dataset)

    #clusterização com fuzzy
    #fuzzy(dataset)

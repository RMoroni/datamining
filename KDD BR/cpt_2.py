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
    train = dataset
    train = train[train['cluster'] != 3]
    #score = dataset[2][0:]
    # amostra = amostra[amostra.silhouette != -2]

    FPC = []
    ERRO = []
    ID = []
    #score=[]
    #test_id=[]
    for id, amostra in train.groupby('scatterplotID'):
        classificacao_column = amostra['cluster']
        classificacao = []
        for classific in classificacao_column:
            classificacao.append(classific)

        sinal_x_column = amostra['signalX']
        sinal_x = []
        for ponto in sinal_x_column:
            sinal_x.append(ponto)

        sinal_y_column = amostra['signalY']
        sinal_y = []
        for ponto in sinal_y_column:
            sinal_y.append(ponto)

        data = np.array([sinal_x, sinal_y])

        init = []  # centroide

        # separa os clusters para descobrir o centro de cada um
        cluster_x = amostra[amostra['cluster'] == 0]
        cluster_y = amostra[amostra['cluster'] == 1]
        cluster_xy = amostra[amostra['cluster'] == 2]

        # para descobrir quantos clusters será utilizado no kmeans
        # certeza tem jeito mais simples de fazer isso, mas funcionou então ok...
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

        # cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data_teste, n_clusters, 2, error=0.005, maxiter=2, init=None)
        u, _, _, _, _, fpc = fuzz.cluster.cmeans_predict(data, init, 2, error=0.005, maxiter=1000)
        '''print('FPC:')
        print(fpc)'''
        FPC.append(fpc)

        if n_clusters == 2:
            # guarda valores únicos
            val, _ = np.unique(classificacao, return_counts=True)
            val = np.sort(val)
            classificacao = np.where(classificacao == val[0], 0, classificacao)
            classificacao = np.where(classificacao == val[1], 1, classificacao)
        elif n_clusters == 1:
            val, _ = np.unique(classificacao, return_counts=True)
            classificacao = np.where(classificacao == val[0], 0, classificacao)

        valores = []
        for i in range(len(classificacao)):
            if u[classificacao[i]][i] < 0.4:
                valores.append(1 - u[classificacao[i]])
        media = 0
        if len(valores) > 0:
            media = np.mean(valores)
            '''if len(u[0]) < 360 & len(u[0] > 340):
                print(media * 1.2)
            elif len(u[0] < 340):
                print(media * 1.4)
            else:
                print(media)'''
        else:
            pass

        #plot_score = score[score['scatterplotID'] == id].score.values[0]
        #print('Clusters - ', n_clusters)
        #print('Score: ',plot_score)
        #print('Pred Score - ',1 - media)
        #score.append(1 - media)
        #test_id.append(id)
        ERRO.append(media)
        ID.append(id)
        #se for aplicar em regressão ou redes neurais imagino q o peso possa ser algo simples como:
        #erro * (objetos mal classficados/total de objetos da amostra)
        #media * (len(valores)/len(classificacao))
    #submission(test_id, score)
    #train['media'] = ERRO
    #dataset[0] = train
    '''score['fuzzy'] = FPC
    dataset[2] = score
    linear_regression_plot(dataset)'''
    return [FPC, ERRO, ID]

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

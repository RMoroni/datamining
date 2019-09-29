import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
 
#dataset_path = '/home/rodrigo/Documents/kddbr-2019/public/'
dataset_path = '/home/viviane/Documents/kddbr-2019/public/'
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
 
def data_treat(dataset):
    #por enquanto vazio, mas aqui pode ser feito tratamentos
    dataset[0] = data_map(dataset[0])
    dataset[1] = data_map(dataset[1])
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
        feat = sample.silhouette.mean()*0.1
        feat = feat - sample.silhouette.std()
        x.append(feat) #feature
        y.append(score[score['scatterplotID'] == id].score.values[0]) #score do scatterplot
        feat = 0.0

    x = np.array(x).reshape((-1, 1)) #tem que fazer isso aqui, sei lá pq
    y = np.array(y)

    model = LinearRegression().fit(x, y) #gera o modelo de regressão linear
    r_sq = model.score(x, y) #calcular a taxa de correlação (printada depois)
    print(r_sq)

    #predição de 4 scatterplot (apenas exemplo, mas é isso que vai para o submission)
    y_pred = model.predict(x[2:6])
    print(y_pred)

def linear_regression_plot(dataset):
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
       #x.append(sample.silhouette.mean())
        feat = feat - (abs(sample.silhouette.mean()-sample.silhouette.median()) + sample.silhouette.std())
        x.append(feat)
        y.append(score[score['scatterplotID'] == id].score.values[0])
        feat = 1.0

    #basicamente, se a feat estiver 'boa', os dados poderão ser aproximados por uma reta
    plt.scatter(x, y)
    plt.show()

def pca(dataset):
    train = dataset[0]
    x = train[['signalX', 'signalY']]
    x = StandardScaler().fit_transform(x)
 
    #PCA ocorre aqui
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    df = pd.DataFrame(data = principalComponents, columns = ['pca1', 'pca2'])
 
    colors = ['red', 'green', 'blue', 'purple']
    plt.scatter(df['pca1'], df['pca2'], c=train['cluster'], cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()

def silhouette_statistic(dataset):
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

def correlation(dataset):
    train = dataset[0][0:]
    score = dataset[2]

    #retira com label L
    train = train[train['cluster'] != 3]

    #adiciona o score para ficar no mesmo DF
    train['score'] = score['score']
    #caso outra feat seja adiciona, é só repetir o processo acima

    #retiro duas colunas (axis=1 -> remove coluna inteira)
    train.drop(['datapointID', 'sampletype'], axis=1)

    #imprime a correlação entre os dados no DF: -1 (anti-correlação) a 1 (correlação perfeita)
    print(train.corr())

def k_means(dataset):
    '''
    Algumas possibilidades:
        * esquecer que isso existiu
        * tentar melhorar isso aqui e usar como previsor
        * (também) utilizar o vetor acurr para formar o x na regressão linear
    '''
    train = dataset[0][0:3840]
    score = dataset[2]
    train = train[train['cluster'] != 3]
    
    acurr = [] #vou salvar aqui a 'taxa' de acerto pra utilizar depois como probabilidade de 'erro'
    #pra criar um 'score individual' é preciso colocar mais uma coluna no train['score_indi'] = val
    #e adicionar se acertou e errou, depois alterar para considerar outros aspectos...
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

        #como nem todos scatter tem todas classes, n_clusters será o max + 1 (essa conta tá errada na vdd)
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
 
    #onde serão montados os gráficos
    #show_plots(dataset)

    #mostra a frequencia dos clusters para amostras com alto score e baixo score
    #frequency_clusters(dataset)

    #mostra anomalias
    #anomalia_clusters(dataset)

    #montei essa função pra tentar deixar 'linear' antes de passar para regressão
    #linear_regression_plot(dataset)

    #pca -> mantem o mesmo número de dimensões, mas 'ajusta' de forma que o padrão entre os samples fiquem 'iguais'
    #apesar de não fazer sentido na questão do pca (por manter as mesmas dimensões), o 'ajuste' mostra que os samples são muito parecidos
    #pca(dataset)

    #regressão linear
    #linear_regression(dataset)

    #correlação
    #correlation(dataset)
    
    #
    #silhouette_statistic(dataset)

    k_means(dataset)
import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
 
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
    scr = []

    for id, sample in train.groupby('scatterplotID'):
        mean.append(sample.silhouette.mean())
        median.append(sample.silhouette.median())
        std.append(sample.silhouette.std())
        scr.append(score[score['scatterplotID'] == id].score.values[0])

    dict = {'score':scr, 'mean':mean, 'median':median, 'std':std}

    df = pd.DataFrame(dict)
    df.to_csv(dataset_path + 'silhouette_statistic.csv')

def correlation(dataset):
    pass

def k_means(dataset):
    train = dataset[0][0:]
    X = []
    kmeans = KMeans(n_clusters=2)
    for _, sample in train.groupby('scatterplotID'):
        x=[]
        x.append(sample.silhouette.mean())
        x.append(sample.silhouette.std())
        X.append(x)

    X = np.array(X)
    y_kmeans = kmeans.fit_predict(X)
    plt.scatter(X[:,0], X[:,1], c=y_kmeans, cmap='viridis')
    plt.show()
    
    wcss = []
    for n in range(2, 8):
        kmeans = KMeans(n_clusters=n)
        kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)

    plt.scatter(wcss, range(2,8))
    plt.show()

    x1, y1 = 2, wcss[0]
    x2, y2 = 8, wcss[len(wcss)-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    print(distances.index(max(distances)) + 2)
    #plt.plot(distances)
    #plt.show()



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
    linear_regression_plot(dataset)

    #pca -> mantem o mesmo número de dimensões, mas 'ajusta' de forma que o padrão entre os samples fiquem 'iguais'
    #apesar de não fazer sentido na questão do pca (por manter as mesmas dimensões), o 'ajuste' mostra que os samples são muito parecidos
    #pca(dataset)

    #regressão linear
    #linear_regression(dataset)

    #
    #silhouette_statistic(dataset)

    #k_means(dataset)
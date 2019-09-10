import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
 
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
 
def pca(data):
    x = data[['signalX', 'signalY']]
    x = StandardScaler().fit_transform(x)
 
    #PCA ocorre aqui
    pca = PCA(n_components=1)
    principalComponents = pca.fit_transform(x)
    df = pd.DataFrame(data = principalComponents, columns = ['pca1'])
 
    colors = ['red', 'green', 'blue', 'purple']
    #plt.scatter(df['pca1'], df['pca2'], c=data['cluster'], cmap=matplotlib.colors.ListedColormap(colors))
    plt.plot(df, 'o', color='black')
    plt.show()

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

def anomalia_clusters(dataset):
    #seleciona os clusters
    cluster_x = dataset[0][dataset[0]['cluster'] == 0]
    cluster_y = dataset[0][dataset[0]['cluster'] == 1]

    #poderia aproveitar da função anterior, mas só pra ñ misturar...
    #alto_score = dataset[2][dataset[2]['score'] >= 0.9]
    baixo_score = dataset[2][dataset[2]['score'] <= 0.1]
    
    #seleciona apenas com baixo score (para cada cluster)
    cluster_x = cluster_x[cluster_x['scatterplotID'].isin(baixo_score['scatterplotID'])]
    cluster_y = cluster_y[cluster_y['scatterplotID'].isin(baixo_score['scatterplotID'])]

    plt.boxplot(cluster_x['signalX'])
    plt.title('Box Plot Cluster X - signal X (score < 0.1))')
    plt.show()

    plt.boxplot(cluster_x['signalY'])
    plt.title('Box Plot Cluster X - signal Y (score < 0.1))')
    plt.show()

    plt.boxplot(cluster_y['signalX'])
    plt.title('Box Plot Cluster Y - signal X (score < 0.1))')
    plt.show()

    plt.boxplot(cluster_y['signalY'])
    plt.title('Box Plot Cluster Y - signal Y (score < 0.1))')
    plt.show()

def linear_regression(dataset):
    train = dataset[0][0:]
    score = dataset[2]

    #retiro os marcados como L
    train = train[train['cluster'] != 3]

    x = []
    y = []

    #regressão linear da média da silhouette de cada plot com o score
    for id, sample in train.groupby('scatterplotID'):
        x.append(sample.silhouette.mean())
        y.append(score[score['scatterplotID'] == id].score.values[0])

    x = np.array(x).reshape((-1, 1))
    y = np.array(y)
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    print(r_sq)

if __name__ == "__main__":
    
    #define parte do csv que será carregada
    #dessa forma é possível fazer o carregamento e processamento por partes
    skiprows = 0 #Pula nenhuma linha
    #skiprows = range(1,384) #Ou seja ignora as linhas de 1 a 385 (preciso da linha 0 p/ colunas)
    nrows = 7680 #Quantas linhas serão carregadas
 
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

    #regressão linear
    linear_regression(dataset)

    #quantos itens para cada amostra
    #print(dataset[0]['scatterplotID'].value_counts())
 
    #clusters X, Y e XY no dataset de treino
    '''cluster_x = dataset[0][dataset[0]['cluster'] == 'X']
    cluster_y = dataset[0][dataset[0]['cluster'] == 'Y']
    cluster_xy = dataset[0][dataset[0]['cluster'] == 'XY']
    cluster_l = dataset[0][dataset[0]['cluster'] == 'L']
 
    #imprime o resultados das contagens
    print("Clusters\n",len(cluster_x),"são X\n",len(cluster_y),"são Y\n",len(cluster_xy),"são XY\n",len(cluster_l),"são L\n")
 
    #imprime o menor valor da coluna silhouette
    print("Silhouette\nMenor valor ->",dataset[0]['silhouette'].min())
 
    #imprime o maior valor da coluna silhouette
    print("Maior valor ->",dataset[0]['silhouette'].max())'''
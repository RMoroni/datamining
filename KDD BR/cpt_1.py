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
        colors = ['red', 'green', 'blue', 'purple']
        plt.scatter(sample['signalX'], sample['signalY'], c=sample['cluster'], cmap=matplotlib.colors.ListedColormap(colors))

        #pego a silhouette e o score do plot
        plot_score = score[score['scatterplotID'] == id].score.values[0]
        silhouette = train[train['scatterplotID'] == id].silhouette

        #com isso é possivel analisar a relação entre a silhouette e o score
        #(onde o score é mais baixo a silhouette se 'espalha' pelo histograma)
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

    #plota o histograma para verificar se existem diferenças nas frequencias de clusters em cada caso
    plt.hist(frames_alto_score['cluster'])
    plt.title('Frequência por Cluster (score >= 0.9)')
    plt.show()

    plt.hist(frames_baixo_score['cluster'])
    plt.title('Frequência por Cluster (score <= 0.1)')
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

def cria_feature_silhouette(dataset):
    silhouette_amostra = []
    for id_amostra in dataset[2]['scatterplotID']:
        objetos = dataset[0][dataset[0]['scatterplotID'] == id_amostra]
        # retira os valores de silhueta iguais a -2
        objetos = objetos[objetos.silhouette != -2]
        # calcula a média das silhuetas dos objetos da amostra
        silhouette_amostra.append(objetos.loc[:, "silhouette"].mean())

    #insere a nova feature no dataframe
    dataset[2]['silhouette'] = silhouette_amostra #

#classifica as amostras de acordo com o score
def cria_feature_classificacao(dataset):
    dataset[2].loc[dataset[2]['score'] <= 0.25, 'classificacao'] = 'pessimo'
    dataset[2].loc[(dataset[2]['score'] > 0.25) & (dataset[2]['score'] <= 0.5), 'classificacao'] = 'ruim'
    dataset[2].loc[(dataset[2]['score'] > 0.5) & (dataset[2]['score'] <= 0.75), 'classificacao'] = 'medio'
    dataset[2].loc[dataset[2]['score'] > 0.75, 'classificacao'] = 'bom'

def cria_feature_qntdadeXY(dataset):
    XY_amostra = []
    for id_amostra in dataset[2]['scatterplotID']:
        objetos = dataset[0][dataset[0]['scatterplotID'] == id_amostra]
        #conta os clusters XY presentes na amostra
        XY_amostra.append(len(objetos['cluster'] == 'XY'))

    # insere a nova feature no dataframe
    dataset[2]['XY'] = XY_amostra

def analise_silhouette_score(dataset):
    #filtra as amostras com determinada classificação (bom, medio, ruim, pessimo)
    objetos = dataset[2][dataset[2]['classificacao'] == 'ruim']
    #ordena as amostras pelo score
    objetos_ordenado = objetos.sort_values(['score'])

    #plota um gráfico de linha
    plt.plot(objetos_ordenado['score'], objetos_ordenado['silhouette'])
    plt.title('Silhouette x Score')
    plt.show()

def analise_XY_score(dataset):
    # filtra as amostras com determinada classificação (bom, medio, ruim, pessimo)
    objetos = dataset[2][dataset[2]['classificacao'] == 'pessimo']
    # ordena as amostras pelo score
    objetos_ordenado = objetos.sort_values(['score'])

    # plota um gráfico de linha
    plt.plot(objetos_ordenado['score'], objetos_ordenado['XY'])
    plt.title('XY x Score')
    plt.show()

if __name__ == "__main__":
    
    #define parte do csv que será carregada
    #dessa forma é possível fazer o carregamento e processamento por partes
    skiprows = 0 #Pula nenhuma linha
    #skiprows = range(1,384) #Ou seja ignora as linhas de 1 a 385 (preciso da linha 0 p/ colunas)
    nrows = None #Quantas linhas serão carregadas
 
    #carrega o dataset
    dataset = load_dataset(skiprows, nrows)
 
    #trata o dataset
    #dataset = data_treat(dataset)
 
    #onde serão montados os gráficos
    #show_plots(dataset)

    #mostra a frequencia dos clusters para amostras com alto score e baixo score
    frequency_clusters(dataset)

    #mostra anomalias
    #anomalia_clusters(dataset)

    #quantos itens para cada amostra
    #print(dataset[0]['scatterplotID'].value_counts())

    #cria a feature silhueta para a amostra
    #cria_feature_silhouette(dataset)

    #cria a feature quantidade de cluster XY para a amostra
    # cria_feature_qntdadeXY(dataset)

    #cria a feature que classifica as amostras de acordo com o score
    #cria_feature_classificacao(dataset)

    #faz análise entre a silhueta e o score da amostra
    #analise_silhouette_score(dataset)

    #faz análise entre a quantidade de cluster XY e o score da amostra
    # analise_XY_score(dataset)
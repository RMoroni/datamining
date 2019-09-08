import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
 
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
    
def knn(data):
    pass
 
def show_plots(data):
    #no primeiro array, pega os elementos de 0 até 384
    train = data[0][0:] #[384:768] e assim por diante...
    score = data[2]

    #Boxplot do X
    #plt.boxplot(train['signalX'])
    #plt.show()
 
    #Boxplot do Y
    #plt.boxplot(train['signalY'])
    #plt.show()
 
    #Boxplot da silhu
    #plt.boxplot(train['silhouette'])
    #plt.show()

    #boxplot do score
    plt.boxplot(score['score'])
    plt.show()
 
    #Histograma do score
    plt.hist(score['score'])
    plt.show()

    #gráfico em barra do X e Y
    #plt.bar(train['signalX'], train['signalY'])
    #plt.show()
 
    #esse trecho ajuda a mostrar as diferenças entre cada amostra
    for _, sample in train.groupby('scatterplotID'):
        #gráfico de X e Y com labels (seguindo as cores)
        colors = ['red', 'green', 'blue', 'purple']
        plt.scatter(sample['signalX'], sample['signalY'], c=sample['cluster'], cmap=matplotlib.colors.ListedColormap(colors))
        plt.show()
 
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
    show_plots(dataset)
 
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
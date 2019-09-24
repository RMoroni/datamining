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

def k_means(dataset):
    train = dataset[0][0:]
    X = []
    kmeans = KMeans(n_clusters=2) #dois clusters (inicial)
    for _, sample in train.groupby('scatterplotID'):
        x=[]
        #duas features, média e desvio padrão
        x.append(sample.silhouette.mean())
        x.append(sample.silhouette.std())
        X.append(x)

    X = np.array(X)
    y_kmeans = kmeans.fit_predict(X) #faz a predição

    #plota gráfico com os clusters
    plt.scatter(X[:,0], X[:,1], c=y_kmeans, cmap='viridis')
    plt.show()
    
    #método do cotovelo
    wcss = []
    for n in range(2, 8):
        #faz kmeans para n de 2 a 8
        kmeans = KMeans(n_clusters=n)
        kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)

    #plota o gráfico dos valores de N
    plt.scatter(wcss, range(2,8))
    plt.show()

    #calcula o melhor N
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

    #k-means
    k_means(dataset)
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

dataset_path = '/home/rodrigo/Documents/kddbr-2019/public/'
#dataset_path = '/home/viviane/Documents/kddbr-2019/public/'
#dataset_path = 'C:'

def load_dataset(skiprows, nrows):
    #abre o csv de treino e teste no caminho específicado
    train = pd.read_csv(dataset_path + 'training_dataset.csv', skiprows=skiprows, nrows=nrows)
    test = pd.read_csv(dataset_path + 'test_dataset.csv', skiprows=skiprows, nrows=nrows)

    #transforma em um único array e retorna
    full_data = [train, test]
    return full_data

def data_map(data):
    #Mapeia o valor de uma string para um int
    cluster_map = {'X':0, 'Y':1, 'XY':2, 'L':3}
    data['cluster'] = data['cluster'].map(cluster_map)
    return data

def plot(data):
    #Utiliza o signalX como eixo X e signalY como eixo Y
    plt.plot(data['signalX'], data['signalY'], 'o', color='black')

    #mostra na tela
    plt.show()

def boxplot(data):
    plt.boxplot(data['signalX'])
    plt.show()
    plt.boxplot(data['signalY'])
    plt.show()

def hist_plot(data):
    pass

def bar_plot(data):
    plt.bar(data['signalX'], data['signalY'])
    plt.show()

def color_bar(data):
    colors = ['red', 'green', 'blue', 'purple']
    plt.scatter(data['signalX'], data['signalY'], c=data['cluster'], cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()

if __name__ == "__main__":

    #define parte do csv que será carregada
    #dessa forma é possível fazer o carregamento e processamento por partes
    #skiprows = 0 #Pula nenhuma linha
    skiprows = range(1,385) #Ou seja ignora as linhas de 1 a 385 (preciso da linha 0 p/ colunas)
    nrows = 384 #Quantas linhas serão carregadas
    #carrega o dataset
    dataset = load_dataset(skiprows, nrows)

    #no primeiro array, pega os elementos de 0 até 384
    sample = dataset[0][0:384] #[384:768] e assim por diante...

    #plota o gráfico
    plot(sample)

    bar_plot(sample)

    boxplot(sample)

    sample = data_map(sample)
    color_bar(sample)

    #clusters X, Y e XY no dataset de treino
    cluster_x = dataset[0][dataset[0]['cluster'] == 'X']
    cluster_y = dataset[0][dataset[0]['cluster'] == 'Y']
    cluster_xy = dataset[0][dataset[0]['cluster'] == 'XY']
    cluster_l = dataset[0][dataset[0]['cluster'] == 'L']

    #imprime o resultados das contagens
    print("Clusters\n",len(cluster_x),"são X\n",len(cluster_y),"são Y\n",len(cluster_xy),"são XY\n",len(cluster_l),"são L\n")

    #imprime o menor valor da coluna silhouette
    print("Silhouette\nMenor valor ->",dataset[0]['silhouette'].min())

    #imprime o maior valor da coluna silhouette
    print("Maior valor ->",dataset[0]['silhouette'].max())

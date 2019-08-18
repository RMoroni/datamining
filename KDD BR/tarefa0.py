import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_path = '/home/rodrigo/Documents/kddbr-2019/public/'

def load_dataset():
    #abre o csv de treino e teste no caminho específicado
    train = pd.read_csv(dataset_path + 'training_dataset.csv')
    test = pd.read_csv(dataset_path + 'test_dataset.csv')

    #transforma em um único array e retorna
    full_data = [train, test]
    return full_data

def plot(data):
    #Utiliza o signalX como eixo X e signalY como eixo Y
    plt.plot(data['signalX'], data['signalY'], 'o', color='black');

    #mostra na tela
    plt.show()

if __name__ == "__main__":

    #carrega o dataset
    dataset = load_dataset()

    #no primeiro array, pega os elementos de 0 até 384
    sample = dataset[0][0:384] #[384:768] e assim por diante...

    #plota o gráfico
    plot(sample)

    #conta quantos clusters X, Y e XY existem no dataset de treino (duas formas)
    cluster_x = 0
    cluster_y = 0
    cluster_xy = 0
    cluster_l = 0

    cluster_x = len(dataset[0][dataset[0]['cluster'] == 'X'])
    cluster_y = len(dataset[0][dataset[0]['cluster'] == 'Y'])
    cluster_xy = len(dataset[0][dataset[0]['cluster'] == 'XY'])
    cluster_l = len(dataset[0][dataset[0]['cluster'] == 'L'])

    '''for item in dataset[0]['cluster']:
        if item == 'X':
            cluster_x += 1
        elif item == 'Y':
            cluster_y += 1
        elif item == 'XY':
            cluster_xy += 1
        else:
            cluster_l += 1'''

    #imprime o resultados das contagens
    print("Clusters\n",cluster_x,"são X\n",cluster_y,"são Y\n",cluster_xy,"são XY\n",cluster_l,"são L\n")

    #imprime o menor valor da coluna silhouette
    print("Silhouette\nMenor valor ->",dataset[0]['silhouette'].min())

    #imprime o maior valor da coluna silhouette
    print("Maior valor ->",dataset[0]['silhouette'].max())

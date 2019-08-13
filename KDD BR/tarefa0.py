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

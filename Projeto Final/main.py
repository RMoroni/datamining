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

dataset_path = '/home/rodrigo/Documents/dados_abertos_urbs/'
#dataset_path = '/home/viviane/Documents/dados_abertos_urbs/'
#dataset_path = 'C:'

def load_dataset(skiprows=0, nrows=None):
    entrada = pd.read_csv(dataset_path + 'entrada_cartao.csv', skiprows=skiprows, nrows=nrows, encoding='latin-1')
    cartao = pd.read_csv(dataset_path + 'cartao.csv', skiprows=skiprows, nrows=nrows)
    bairro = pd.read_csv(dataset_path + 'bairro.csv')
    full_data = [entrada, cartao, bairro]
    return full_data

def data_map(dataset):
    #cluster_map = {'X':0, 'Y':1, 'XY':2, 'L':3}
    #data['cluster'] = data['cluster'].map(cluster_map)
    return data

def all_plots(dataset):
    pass

if __name__ == "__main__":
    print('Carregando os dados...')
    dataset = load_dataset()
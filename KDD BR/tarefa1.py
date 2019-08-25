import requests
from bs4 import BeautifulSoup

url="https://www.amazon.com.br/Cafeteira-Francesa-Hamilton-Beach-Pequeno/dp/B00E5613M4"

req = requests.get(url)

soup = BeautifulSoup(req.content, 'html.parser')

nomeProduto = soup.find(id="productTitle")

print('Nome do Produto: ' + nomeProduto.text)

precoProduto = soup.find(id="price_inside_buybox")

print('Preço do Produto: ' + precoProduto.text)

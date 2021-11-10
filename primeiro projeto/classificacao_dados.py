#esta importanto a biblioteca para passar o link para trazer a tabela
import pandas as pd
#link da tabela 
uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
#devolve a tabela do link 
dados = pd.read_csv(uri)
#impimir só as primeiras linhas
dados.head()

#trocando o nome das colunas em ingles para portugues
mapa = {
     "home" : "principal",
     "how_it_works" : "como_funciona",
     "contact" : "contato",
     "bought" : "comprado"
} 
dados = dados.rename(columns = mapa)

x = dados[["principal", "como_funciona", "contato"]]
x.head()
y = dados["comprado"]
y.head()

#treinando os dados apartir de uma biblioteca
#importanto uma biblioteca de treino e texte
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#vai sempre replicar o mesmo teste, sendo assim não vai colocar testes aleatorios
SEED = 20

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, 
                                                          random_state = SEED, test_size = 0.25,
                                                          stratify = y)
#strarify ela vai separar proporcionalmente de acordo com y
print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y) 

#testando
previsoes = modelo.predict(teste_x)

#comparando as previsoes com o teste_y
taxa_de_acerto = accuracy_score(teste_y, previsoes) * 100
print("a taxa de acertos foi %.2f%%" % taxa_de_acerto)
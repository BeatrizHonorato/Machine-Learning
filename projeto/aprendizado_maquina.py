# features (1 sim, 0 não)
# pelo longo?
# perna curta?
# faz auau?
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

#1 => porco, 0 => cachorro

treino_x = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
treino_y = [1, 1, 1, 0, 0, 0] #label / #etiqueta  

from sklearn.svm import LinearSVC
model = LinearSVC()
model.fit(treino_x,treino_y)

misterio1 = [1, 1, 1]
misterio2 = [1, 1, 0]
misterio3 = [0, 1, 1]
teste_x = [misterio1, misterio2, misterio3]
teste_y = [0, 1, 1]
#model.predict(teste)

previsoes = model.predict(teste_x)

corretos = (previsoes == teste_y).sum()
total = len(teste_x)
taxa_acertos = corretos/total
print("Taxa de acertos: ", taxa_acertos * 100)

from sklearn.metrics import accuracy_score
taxa_acertos = accuracy_score(teste_y, previsoes)
print("Taxa de acertos: ",taxa_acertos * 100)
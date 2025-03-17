from sklearn.metrics import confusion_matrix
#Usado para separação entre teste e treinamento
from sklearn.model_selection import train_test_split
# Dataset prontos
from sklearn import datasets
#É um modelo em que as camadas são empilhadas sequencialmente. 
# É um modelo simples onde cada camada tem exatamente uma entrada e uma saída.
from keras.api.models import Sequential
# Representa uma camada totalmente conectada, 
# onde cada neurônio de uma camada está conectado a todos os neurônios da camada anterior.
from keras.api.layers import Dense
#converte rótulos de classe em um formato one-hot 
from keras.api.utils import to_categorical
import numpy as np
# Ler o arquivo
dados = datasets.load_iris()

#Matriz
previsores = dados.data
classes = dados.target

# converte rótulos de classe em um formato one-hot. 
# Esse formato é comumente utilizado em tarefas de classificação, 
# especialmente em redes neurais, onde a saída para cada exemplo de treinamento é representada por um vetor binário.
#One-hot encoding é um processo em que cada valor de uma classe é representado por um vetor 
# com um único valor 1 em uma posição correspondente à classe e 0 nas outras posições
classe_dummy = to_categorical(classes, num_classes=3)
print(classe_dummy)
# Divisão entre treinamento e teste(30% para testar e 70% para treinar)
x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                   classe_dummy,
                                                                   test_size=0.3,
                                                                   random_state=0
                                                                   )

#Estrutura de rede neural com a classe Sequential
modelo = Sequential()

#Primeira camada oculta, 5 neuronios, 4 neuronios de entrada.
# parâmetro input_dim=4 é utilizado para especificar o número de entradas (features) em uma rede neural, 
# geralmente na camada de entrada.
#Dense: Refere-se a uma camada densa, onde cada neurônio está conectado a todos os neurônios da camada anterior.
#units=5: Isso indica que a camada terá 5 neurônios
modelo.add(Dense(units=5, input_dim=4))

#Segunda camada oculta
modelo.add(Dense(units=4))

#Terceira camada oculta
#Função softmax porque temos um problema de classificação com mais de duas classes(è gerado uma probabilidade em cada neuronio)
modelo.add(Dense(units= 3, activation='softmax'))


print(modelo.summary())

# Configuração de parametros da rede neural(adam= algoritmo para atualizar os pesos e loss= cálcular erro)
# O Adam: (Adaptive Moment Estimation) é um dos otimizadores mais populares e eficientes, 
# pois ajusta automaticamente a taxa de aprendizado com base nas primeiras e segundas derivadas da função de perda. 
# Ele combina as vantagens de outros otimizadores, como o SGD (gradiente descendente estocástico) e o RMSprop.
# loss=categorical_crossentrop: é a função de perda usada para classificação multiclasse. Ela é usada quando as saídas são codificadas de forma one-hot (ou seja, cada classe é representada por um vetor onde apenas uma posição é 1 e as outras são 0).
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Treinamento, dividindo a base de treinamento em  uma porção para validação(validation_data)
modelo.fit(x_treinamento, y_treinamento, epochs=1000, validation_data=(x_teste, y_teste))
previsoes = modelo.predict(x_teste)
# (previsoes > 0.5): Compara as previsões com 0.5. Se o valor for maior que 0.5,
# a previsão será considerada como "1" (classe positiva), caso contrário, será "0" (classe negativa).
previsoes = (previsoes > 0.5)
print(previsoes)

#Buscando a posição com maior valor
y_teste_matrix = [np.argmax(t) for t in y_teste]
x_previsao_matrix = [np.argmax(t) for t in previsoes]


#Geração de matrix de confusão
confusao = confusion_matrix(y_teste_matrix, x_previsao_matrix)
print(f'Matriz de Confusão: {confusao}')
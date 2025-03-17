
# Rede Neural Keras para Classificação com o Dataset Iris

Este projeto utiliza a biblioteca Keras para construir e treinar uma rede neural simples com o objetivo de classificar as espécies de flores no conjunto de dados Iris. O modelo é treinado utilizando um conjunto de dados de treinamento e avaliado em um conjunto de teste. O desempenho do modelo é avaliado com a matriz de confusão.

## Bibliotecas
- Keras
- Scikit-learn
- Numpy

## Estrutura do Código

1. **Carregamento do Dataset**: O conjunto de dados Iris é carregado utilizando o `datasets.load_iris()` do Scikit-learn.
2. **Pré-processamento de Dados**:
   - Os dados são divididos em variáveis independentes (previsores) e a variável dependente (classe).
   - A variável dependente (classes) é convertida em formato one-hot encoding utilizando o `to_categorical` do Keras.
   - Os dados são divididos em conjuntos de treinamento e teste utilizando a função `train_test_split` do Scikit-learn.
3. **Criação do Modelo**:
   - Um modelo simples de rede neural é criado com a classe `Sequential` do Keras.
   - A primeira camada oculta tem 5 neurônios e 4 entradas.
   - A segunda camada oculta tem 4 neurônios.
   - A terceira camada é a camada de saída, com 3 neurônios (um para cada classe) e a função de ativação `softmax`.
4. **Compilação e Treinamento**:
   - O modelo é compilado utilizando o otimizador 'adam' e a função de perda 'categorical_crossentropy'.
   - O treinamento é realizado por 1000 épocas.
5. **Avaliação do Modelo**:
   - Após o treinamento, o modelo é utilizado para fazer previsões no conjunto de teste.
   - A matriz de confusão é gerada para avaliar o desempenho do modelo.



## Resultados

O modelo treina e avalia a rede neural, gerando a matriz de confusão para avaliar a acurácia do modelo. 
**Accuracy de 0.978**.


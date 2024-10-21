import numpy as np

class Perceptron:
    
    def __init__(self, a, epocas):
        self.a = a  # Taxa de aprendizado / alpha
        self.epocas = epocas  # Número de iterações para treinamento
        self.w = None  # Pesos do modelo, inicializados como None

    def funcaoativacao(self, x):
        # Função de ativação que retorna 1 se x >= 0, caso contrário 0
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        N, D = X.shape  # N: número de amostras, D: número de características
        
        # Inicializa pesos com uma coluna extra para o bias
        self.w = np.zeros(D + 1)  

        # Adiciona uma coluna de 1s para incluir o bias nas entradas
        X_com_1 = np.hstack((np.ones((N, 1)), X))

        for _ in range(self.epocas):
            # Calcula o produto escalar entre as entradas e os pesos
            linear_output = np.dot(X_com_1, self.w) 

            # Aplicar a função de ativação para obter as previsões do modelo (0 ou 1)
            y_pred = self.funcaoativacao(linear_output)

            # Calcula o erro entre as previsões (saída predita) e as saídas reais
            erro = (y - y_pred)

            # Calcula o gradiente do erro em relação aos pesos
            # G = - (somatório(erro_i * X_i) / N) representa a direção e a magnitude
            # em que os pesos devem ser ajustados. 
            grad = - np.dot(X_com_1.T, erro) / (N)

            # Atualizar os pesos do modelo usando a taxa de aprendizado
            self.w -= self.a * grad

    def predict(self, X):
        # Adiciona coluna de 1s para as entradas, para incluir o bias
        X_com_1 = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Calcular a saída linear do modelo: produto escalar entre as entradas e os pesos
        linear_output = np.dot(X_com_1, self.w)
        
        # Aplicar a função de ativação para obter as saídas preditas
        y_pred = self.funcaoativacao(linear_output)
        return y_pred

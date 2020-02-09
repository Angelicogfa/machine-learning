# Regressão linear
# Um modelo linear faz uma previsão calculando uma soma ponderada das caraceristicas de entrada,
# mais uma constante chamada 'termo de polarização (também chamada coeficiente linear).
# O => theta
# y = O + O¹X¹ + O²X² + ... + OnXn

# y => é o valor previsto
# n => é o número de caracteristicas
# X(i) => é o valor da i-ésima caracteristica
# O(j) => é o parâmetro do modelo j (incluindo o termo de polarização O(0) e os pesos das caracteristicas O¹, O², ..., 0n)

# Isso pode ser usada de maneira muito mais concisa usando uma forma vetorial
# y = hO(x) = O(transposta) * x

# O => é o vetor de parâmetros do modelo, que contém o termo de polarização O0 e os pesos das caracteristicas O¹ a On
# O (transposta) => é a transposiçao de O (um vetor de linha vira vetor de coluna)
# x => é o vetor de caracteristicas da instancia, que contém x(0) a x(n) com x(0) sempre igual a 1
# O (transposta) * x => é o produto escalar de O(transposta) e x
# h0 => é a função de hipotese, que utiliza os parâmetros do modelo O

# Função MSE de custo para um modelo de regressão linear
# MSE(X, h0) = 1/m (somatoria m; i=1) (O (transposta) * x^i - y^i)²

# Método dos mínimos quadrados
# Para encontrar o valor de O, que minimiza a função de custo, existe uma função: Método dos minimos quadrados
# Ô = (X (transposta) * X) ^ -1 * X (transposta) * y

# Ô => é o valor de O que minimiza a função de custo
# y => é o vetor dos valores do alvo contendo y^(1) a y^(m)

import numpy as np
import matplotlib.pyplot as plt
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

plt.plot(X, y, 'b.')
plt.ylabel('y')
plt.xlabel('x1')
plt.axis([0,2,0,15])
plt.show()

# Agora calcularemos o Ô usando o método dos minimos quadrados
X_b = np.c_[np.ones((100,1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2,1)), X_new]
y_predict = X_new_b.dot(theta_best)
y_predict

plt.plot(X_new, y_predict, 'r-', linewidth=2, label='Predictions')
plt.plot(X, y, 'b.')
plt.xlabel("$x_1$", fontsize=18)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.legend(loc='upper left', fontsize=14)
plt.axis([0,2,0,15])
plt.show()

# O código acima equivale a este:
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
lin_reg.predict(X_new)

# Gradiente Descendente
# É um algoritmo de otimização muito genérico capaz de encontrar ótimas soluções para uma ampla gama
# de problemas. A ideia geral do GD é ajustar iterativamente os parâmetros para minimizar a função
# de custo.
# O gradiente descendente mede o gradiente local da função de erro em relação ao vetor de parametro
# O, e vai na direção do GD. Quando o gradiente for 0, você atingiu o mínimo.
# Geralmente você começa preenchendo o valor O com valores aleatórios. (isso é chamado de inicialização
# aleatória), e então melhora gradualmente, dando pequenos passos por cada vez, cada passo
# tentanto atingir a  função de custo (por exemplo o MSE), até que o algoritmo convirja para um 
# mínimo.
# O tamanho do passo é um parâmetro importante para o GD, determinado pelo hiperparametro 'taxa de aprendizagem'.
# Se a taxa de aprendizado for muito pequena, demorará muito para convergir. Se for muito alta, você
# pode atravessar o vale e acabar do outro lado.

# Ao utilizar o GD, você deve garantir que todas as caracteristicas tenham uma escala similar (por exemplo,
# utilizando a classe StandardScaler do scikit-learn), ou então demorará muito mais para convergir.

# Para implementar o GD, é preciso calcular o gradiente da função de custo em relação cada parametro
# do modelo de O(j). Em outras palavras, você precisa calcular quanto mudará a função de custo se você modificar
# somente um pouco O(j). Isso é chamado de derivação parcial. 
# A função abaixo calcula a derivada parcial da função de custo em relação ao parametro O(j), notado por
# a / aO(j) MSE (O) => Derivada / Derivada parcial O(j) da função MSE(O)
# a => derivada

# a/aO(j) MSE(O) = 2/m * somatoria m (i..1) (O(transposta) * x^i - y^i) * x(j)^(i)

# Em vez de calcular individualmente essas derivadas parciais, você pode utilizar a equação abaixo para 
# calcula-las de uma vez só. O vetor gradiente, descrito \/O MSE(O), contem todas as derivadas parciais da
# função de custo (uma para cada parametro do modelo).


#               | a/aO(0) MSE(O) |
# \/O MSE(O) =  | a/aO(1) MSE(O) | = 2/m* X(transposta) * (X + O - y)
#               | a/aO(n) MSE(O) |

# Observe que esta formula envolve cálculos em cada etapa do GD sobre o conjunto completo de treinamento X! É
# por isso que o algoritmo é chamado Gradiente Descendente em Lote: ele utiliza todo o lote de dados em cada etapa.
# Como resultado ele é terrivelmente lente em grandes conjuntos. No entanto, o GD se dimensiona bem com a quantidade
# de caracteristicas; treinar um modelo de regressão linear quando há centenas de milhares de caracteristicas, será muito
# mais rapido se utilizarmos o GD do que se utilizarmos o método dos Minimos Quadrados.

# Vejamos uma implementação rápida desse algoritmo

theta_path_bgd = []

def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else "r--"
            plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
        if theta_path is not None:
            theta_path.append(theta)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 2, 0, 15])
    plt.title(r"$\eta = {}$".format(eta), fontsize=16)

np.random.seed(42)
theta = np.random.randn(2,1)  # random initialization

plt.figure(figsize=(10,4))
plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
plt.ylabel("$y$", rotation=0, fontsize=18)
plt.subplot(132); plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
plt.subplot(133); plot_gradient_descent(theta, eta=0.5)

plt.show()

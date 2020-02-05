# Regressão linear
# Um modelo linear faz uma previsão calculando uma soma ponderada das caraceristicas de entrada,
# mais uma constante chamada 'termo de polarização (também chamada coeficiente linear).
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

plt.plot(X_new, y_predict, 'r-')
plt.plot(X, y, 'b.')
plt.ylabel('y')
plt.xlabel('x1')
plt.legend()
plt.axis([0,2,0,15])
plt.show()
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


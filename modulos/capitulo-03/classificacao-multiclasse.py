from sklearn.datasets import load_digits

data, target = load_digits(10, True)

from matplotlib.cm import binary 
import matplotlib.pyplot as plt

# Separação de dados para treino, teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.3)


some_digit = x_train[11]

plt.imshow(some_digit.reshape(8,8), cmap= binary, interpolation="nearest")
y_train[11]

# Classificação multiclasse com o sgdclassifier -> Descendente gradiente estocrastico
from sklearn.linear_model import SGDClassifier
sgdc = SGDClassifier()
sgdc.fit(x_train, y_train)
sgdc.predict([some_digit])

# Classificação multiclasse com classificadores binarios
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(x_train, y_train)
ovo_clf.predict([some_digit])

# Usando random florest
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier()
forest_clf.fit(x_train, y_train)
forest_clf.predict([some_digit])
forest_clf.predict_proba([some_digit])


# Validando o score com validação cruzada
from sklearn.model_selection import cross_val_score
cross_val_score(sgdc, x_train, y_train, cv= 5)
cross_val_score(ovo_clf, x_train, y_train, cv=3)
cross_val_score(forest_clf, x_train, y_train, cv=3)

# Dimensionando as entradas para melhorar a precisão
from sklearn.preprocessing import StandardScaler
import numpy as np
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float))
cross_val_score(sgdc, x_train_scaled, y_train, cv= 5)
cross_val_score(ovo_clf, x_train_scaled, y_train, cv=3)
cross_val_score(forest_clf, x_train_scaled, y_train, cv=3)

# Analisando erros
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
y_train_pred = cross_val_predict(sgdc, x_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

plt.matshow(conf_mx)
plt.show()

## Plotagem de erros por classe
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx /row_sums
np.fill_diagonal(norm_conf_mx, 0)

plt.matshow(norm_conf_mx, cmap= plt.cm.gray)
plt.ylabel('Classes reais')
plt.xlabel('Classes previstas')
plt.title('Matrix de confusão')
plt.show()

# Analisando imagens 3 e 5 para entender o erro
cl_a, cl_b = 3, 5

x_aa = x_train[(y_train == cl_a) & (y_train_pred == cl_a)]
x_ab = x_train[(y_train == cl_a) & (y_train_pred == cl_b)]
x_ba = x_train[(y_train == cl_b) & (y_train_pred == cl_a)]
x_bb = x_train[(y_train == cl_b) & (y_train_pred == cl_b)]

def plot_digit(data, size):
    image = data.reshape(size, size)
    plt.imshow(image, cmap = binary, interpolation="nearest")
    # plt.imshow(image)
    plt.axis("off")

def plot_digits(instances, images_per_row=10, **options):
    size = 8
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = binary, **options)
    plt.axis("off")

plt.figure(figsize=(8,8))

plt.subplot(221); plot_digits(x_aa[:25], images_per_row=5)
plt.subplot(221); plot_digits(x_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(x_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(x_bb[:25], images_per_row=5)
plt.show()

# ---------------------------------------------------------------------------
# Classificação multilabel
## Sistema de classificação que mostra varios rotulos binários é chamado
## de sistema de classificação

from sklearn.neighbors import KNeighborsClassifier

# Gera uma label para os elementos alvo para avaliar se eles são menores que 7
y_train_large = (y_train >= 7)

# Gera uma label para os elementos alvo para avaliar se eles são numeros impares
y_train_odd = (y_train % 2 == 1)

y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train, y_multilabel)

knn_clf.predict([some_digit])

from sklearn.metrics import f1_score
y_train_knn_pred = cross_val_predict(knn_clf, x_train, y_multilabel, cv = 3)
f1_score(y_multilabel, y_train_knn_pred, average='macro')
# f1_score(y_multilabel, y_train_knn_pred, average='weighted')

# ------------------------------------------------------------------------------
# Classificação multioutput
## Criando ruidos em imagem 
noise = np.random.randint(0, 100, (len(x_train), 64))
x_train_mod = x_train + noise
noise = np.random.randint(0, 100, (len(x_test), 64))
x_test_mod = x_test + noise
y_train_mod = x_train
y_test_mod = x_test

knn_clf.fit(x_train_mod, y_train_mod)
clean_digit = knn_clf.predict([x_test_mod[0]])
plot_digit(clean_digit, 8)
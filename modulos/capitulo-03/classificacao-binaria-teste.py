# Problema
# Classificar as plantas em setosa e não setosa
from sklearn.datasets import load_iris
data, target = load_iris(True)

x_data = data
y_data = (target == 0)

len(y_data[y_data == True])
len(y_data[y_data == False])

# Split de dados entre treino e teste

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4)

# Validação cruzada para classificaçao com SGDClassifier
from sklearn.linear_model import SGDClassifier
sgd_class = SGDClassifier()

sgd_class.fit(x_train, y_train)
y_score = sgd_class.predict(x_train)

# from sklearn.model_selection import cross_val_predict, cross_validate
# y_score = cross_val_predict(sgd_class, x_train, y_train, cv=3, n_jobs=-1)

# metricas
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
confusion_matrix(y_train, y_score)
precision_score(y_train, y_score)
recall_score(y_train, y_score)
f1_score(y_train, y_score)

# validando com os dados de testes
y_score_test = sgd_class.predict(x_test)

confusion_matrix(y_test, y_score_test)
precision_score(y_test, y_score_test)
recall_score(y_test, y_score_test)
f1_score(y_test, y_score_test)
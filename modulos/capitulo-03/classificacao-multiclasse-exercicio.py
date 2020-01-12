from skimage.transform import rotate
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

data, target = load_digits(10, True)

# Config
np.random.seed(42)

# train test
x_train, x_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=42)

# -----------------------------------------------------------------------------------

# 1 - Classificador com 97% precisao ou mais


def print_confusion_matrix(pred):
    return pd.DataFrame(pred, columns=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])


# Dummy Classifier para obter uma acuracia minima
dummy = DummyClassifier()
dummy.fit(x_train, y_train)

dummy_predict = dummy.predict(x_train)

accuracy_score(y_train, dummy_predict)
confusion_matrix(y_train, dummy_predict)

# Multilabel classifier
model = OneVsRestClassifier(DecisionTreeClassifier())
model.fit(x_train, y_train)

y_train_predict = model.predict(x_train)
accuracy_score(y_train, y_train_predict)
confusion_matrix(y_train, y_train_predict)

y_test_predict = model.predict(x_test)
accuracy_score(y_test, y_test_predict)

print_confusion_matrix(confusion_matrix(y_test, y_test_predict))

# Decision tree
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_train_predict = model.predict(x_train)
accuracy_score(y_train, y_train_predict)

y_test_predict = model.predict(x_test)
accuracy_score(y_test, y_test_predict)

# Modelo
params = {"n_neighbors": [3, 5, 8, 13], "weights": [
    "uniform", "distance"], "metric": ["euclidean", "manhattan"]}

g_search = GridSearchCV(KNeighborsClassifier(), params,
                        verbose=1, cv=5,  n_jobs=-1)
g_search.fit(x_train, y_train)

g_search.best_params_
g_search.best_score_
model = g_search.best_estimator_

y_model_predict = model.predict(x_test)
accuracy_score(y_test, y_model_predict)
print_confusion_matrix(confusion_matrix(y_test, y_model_predict))

# escalando os dados

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
g_search.fit(x_train_scaled, y_train)

g_search.best_params_
g_search.best_score_
model = g_search.best_estimator_

y_model_predict = model.predict(scaler.transform(x_test))
accuracy_score(y_test, y_model_predict)
print_confusion_matrix(confusion_matrix(y_test, y_model_predict))

# --------------------------------------------------------------------------------------

# Rotacionar imagem
#  !pip install scikit-image

def rotate_image(img_array, degree):
    image = img_array.reshape(8, 8)
    return rotate(image, degree).reshape(64)

def print_image(img_array, color='Greys'):
    image = img_array.reshape(8,8)
    plt.imshow(image, cmap=color)

x_train_augmented = [x for x in x_train]
y_train_augmented = [y for y in y_train]

for position in [90, 180, 270]:
    for image, label in zip(x_train, y_train):
        x_train_augmented.append(rotate_image(image, position))
        y_train_augmented.append(label)

x_train_augmented = np.array(x_train_augmented)
y_train_augmented = np.array(y_train_augmented)

knn_class = KNeighborsClassifier(**g_search.best_params_)
knn_class.fit(x_train_augmented, y_train_augmented)

y_train_augmented_predict = knn_class.predict(x_train_augmented)
accuracy_score(y_train_augmented, y_train_augmented_predict)

y_test_augmented_predict = knn_class.predict(x_test)
accuracy_score(y_test, y_test_augmented_predict)

